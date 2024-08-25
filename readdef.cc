// ped-sim: pedigree simulation tool
//
// This program is distributed under the terms of the GNU General Public License

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <errno.h>
#include <assert.h>
#include "readdef.h"

// TODO: only use sexConstraints array when there are sex-specific maps?
// TODO: make branchNumSpouses positive


// Reads in the pedigree formats from the def file, including the type of the
// pedigree (full, half, or double) and the number of samples to produce in
// every generation
void readDef(vector<SimDetails> &simDetails, char *defFile) {
  // open def file:
  FILE *in = fopen(defFile, "r");
  if (!in) {
    printf("ERROR: could not open def file %s!\n", defFile);
    perror("open");
    exit(1);
  }

  // def file gives the number of samples to print; we store this in a 2d array
  // with the first index being generation number and the second index the
  // branch number
  int **curNumSampsToPrint = NULL;
  // Have variable number of branches in each generation
  int *curNumBranches = NULL;
  // Who are the parents of each branch in each generation?
  // Contains <curNumGen> rows, and <2*curNumBranches[gen]> columns on each row.
  // Stores the generation and branch numbers of the two parents.
  // Negative branch values correspond to founders that are stored in the same
  // branch number as the other parent.
  Parent **curBranchParents = NULL;
  // Gives numerical values indicating dependencies of sex assignments for each
  // branch. For example, if the person in branch 1 has children with the
  // individual in branch 2 and 3, branch 2 and 3 must have the same sex.
  // Also stores the sexes of the i1 individuals when these are specified in
  // the def file.
  SexConstraint **curSexConstraints = NULL;
  // If all i1 individuals are to have the same sex, the following gives its
  // value. A value of -1 corresponds to random assignment.
  // Branch-specific i1 specifications override this
  int curI1Sex = -1;
  // Counts of number of non-founder spouses for each generation/branch
  int **curBranchNumSpouses = NULL;
  int curNumGen = 0;
  // for ensuring generations are in increasing order. This requirement arises
  // from the fact that we assign the number of branches in each generation to
  // be equal to the previous generation (except generation 2), so we need to
  // know which generation we've assigned the generation numbers to and update
  // branch counts for any generations that aren't explicitly listed.
  int lastReadGen = -1;

  // Tracks whether there has been an explicit assignment of the parents of
  // each branch to avoid double assignments and giving default assignments.
  vector<bool> branchParentsAssigned;
  // Stores sets of individuals that are required to have the same and/or
  // opposite sex assignments by virtue of their being spouses.
  // Also, for each set, stores the sex (either 0 or 1 for male or female) of
  // the set if this is specified for any member of a set (or their partners)
  // in the def file
  // (this value is temporary: its contents get put into <curSexConstraints>
  // during processing)
  vector< pair<set<Parent,ParentComp>,int8_t>* > spouseDependencies;

  bool warningGiven = false;

  size_t bytesRead = 1024;
  char *buffer = (char *) malloc(bytesRead + 1);
  if (buffer == NULL) {
    printf("ERROR: out of memory");
    exit(5);
  }
  const char *delim = " \t\n";

  int line = 0;
  while (getline(&buffer, &bytesRead, in) >= 0) {
    line++;

    char *token, *saveptr, *endptr;
    token = strtok_r(buffer, delim, &saveptr);

    if (token == NULL || token[0] == '#') {
      // blank line or comment -- skip
      continue;
    }

    if (strcmp(token, "def") == 0) {
      if (spouseDependencies.size() > 0) {
	// do some bookkeeping to finalize the previously read pedigree
	int lastNumGen = curNumGen;
	finishLastDef(lastNumGen, curSexConstraints, spouseDependencies);
      }

      /////////////////////////////////////////////////////////////////////////
      // parse new pedigree description
      char *name = strtok_r(NULL, delim, &saveptr);
      char *numRepsStr = strtok_r(NULL, delim, &saveptr);
      char *numGenStr = strtok_r(NULL, delim, &saveptr);
      char *i1SexStr = strtok_r(NULL, delim, &saveptr);
      // Note: leaving i1SexStr out of the next conditional because it can be
      //       NULL or non-NULL
      if (name == NULL || numRepsStr == NULL || numGenStr == NULL ||
				      strtok_r(NULL, delim, &saveptr) != NULL) {
	fprintf(stderr, "ERROR: line %d in def: expect four or five fields for pedigree definition:\n",
		line);
	fprintf(stderr, "       def [name] [numReps] [numGen] <sex of i1>\n");
	exit(5);
      }
      errno = 0; // initially
      int curNumReps = strtol(numRepsStr, &endptr, 10);
      if (errno != 0 || *endptr != '\0') {
	fprintf(stderr, "ERROR: line %d in def: expected number of replicates to simulate as second token\n",
		line);
	if (errno != 0)
	  perror("strtol");
	exit(2);
      }
      curNumGen = strtol(numGenStr, &endptr, 10);
      if (errno != 0 || *endptr != '\0') {
	fprintf(stderr, "ERROR: line %d in def: expected number of generations to simulate as third",
		line);
	fprintf(stderr, "      token\n");
	if (errno != 0)
	  perror("strtol");
	exit(2);
      }

      if (i1SexStr == NULL)
	curI1Sex = -1;
      else {
	if (strcmp(i1SexStr, "F") == 0) {
	  curI1Sex = 1;
	}
	else if (strcmp(i1SexStr, "M") == 0) {
	  curI1Sex = 0;
	}
	else {
	  fprintf(stderr, "ERROR: line %d in def: allowed values for sex of i1 field are 'M' and 'F'\n",
		  line);
	  fprintf(stderr, "       got %s\n", i1SexStr);
	  exit(7);
	}
      }

      // TODO: slow linear search to ensure lack of repetition of the pedigree
      // names; probably fast enough
      for(auto it = simDetails.begin(); it != simDetails.end(); it++) {
	if (strcmp(it->name, name) == 0) {
	  fprintf(stderr, "ERROR: line %d in def: name of pedigree is same as previous pedigree\n",
		  line);
	  exit(5);
	}
      }

      curNumSampsToPrint = new int*[curNumGen];
      curNumBranches = new int[curNumGen];
      curBranchParents = new Parent*[curNumGen];
      curSexConstraints = new SexConstraint*[curNumGen];
      curBranchNumSpouses = new int*[curNumGen];
      if (curNumSampsToPrint == NULL || curNumBranches == NULL ||
	  curBranchParents == NULL || curSexConstraints == NULL ||
	  curBranchNumSpouses == NULL) {
	printf("ERROR: out of memory");
	exit(5);
      }
      if (lastReadGen >= 0)
	lastReadGen = -1; // reset

      for(int gen = 0; gen < curNumGen; gen++) {
	// initially
	curNumSampsToPrint[gen] = NULL;
	// set to -1 initially so we know these are unassigned; will update
	// later
	curNumBranches[gen] = -1;
	curBranchParents[gen] = NULL;
	curSexConstraints[gen] = NULL;
	curBranchNumSpouses[gen] = NULL;
      }
      simDetails.emplace_back(curNumReps, curNumGen, curNumSampsToPrint,
			      curNumBranches, curBranchParents,
			      curSexConstraints, curI1Sex,
			      curBranchNumSpouses, name);
      continue;
    }

    ///////////////////////////////////////////////////////////////////////////
    // parse line with information about a generation in the current pedigree

    // is there a current pedigree?
    if (curNumSampsToPrint == NULL) {
      fprintf(stderr, "ERROR: line %d in def: expect four or five fields for pedigree definition:\n",
	      line);
      fprintf(stderr, "       def [name] [numReps] [numGen] <sex of i1>\n");
      exit(5);
    }

    char *genNumStr = token;
    char *numSampsStr = strtok_r(NULL, delim, &saveptr);
    char *branchStr = strtok_r(NULL, delim, &saveptr);

    errno = 0; // initially
    int generation = strtol(genNumStr, &endptr, 10);
    if (errno != 0 || *endptr != '\0') {
      fprintf(stderr, "ERROR: line %d in def: expected generation number or \"def\" as first token\n",
	  line);
      if (errno != 0)
	perror("strtol");
      exit(2);
    }

    if (numSampsStr == NULL) {
      printf("ERROR: improper line number %d in def file: expected at least two fields\n",
	      line);
      exit(5);
    }
    int numSamps = strtol(numSampsStr, &endptr, 10);
    if (errno != 0 || *endptr != '\0') {
      fprintf(stderr, "ERROR: line %d in def: expected number of samples to print as second token\n",
	  line);
      if (errno != 0)
	perror("strtol");
      exit(2);
    }

    if (generation < 1 || generation > curNumGen) {
      fprintf(stderr, "ERROR: line %d in def: generation %d below 1 or above %d (max number\n",
	      line, generation, curNumGen);
      fprintf(stderr, "       of generations)\n");
      exit(1);
    }
    if (numSamps < 0) {
      fprintf(stderr, "ERROR: line %d in def: in generation %d, number of samples to print\n",
	      line, generation);
      fprintf(stderr, "       below 0\n");
      exit(2);
    }
    if (generation == 1 && numSamps > 1) {
      fprintf(stderr, "ERROR: line %d in def: in generation 1, if founders are to be printed must\n",
	      line);
      fprintf(stderr, "       list 1 as the number to be printed (others invalid)\n");
      exit(2);
    }

    if (generation <= lastReadGen) {
      fprintf(stderr, "ERROR: line %d in def: generation numbers must be in increasing order\n",
	      line);
      exit(7);
    }

    // if <curNumBranches> != -1, have prior definition for generation.
    // subtract 1 from <generation> because array is 0 based
    if (curNumBranches[generation - 1] != -1) {
      fprintf(stderr, "ERROR: line %d in def: multiple entries for generation %d\n",
	      line, generation);
      exit(2);
    }
    // Will assign <numSamps> to each branch of this generation below -- first
    // need to know how many branches are in this generation

    // Assign number of branches (and parents) for generations that are not
    // explicitly listed. In general the number of branches is equal to the
    // number in the previous generation. The exceptions are generation 1 which
    // defaults to 1 branch, and generation 2 which defaults to 2 branches when
    // generation 1 has only 1 branch (otherwise it's assigned the same as the
    // previous generation)
    for(int i = lastReadGen + 1; i < generation - 1; i++) {
      if (i == 0)
	curNumBranches[0] = 1;
      else if (i == 1 && curNumBranches[0] == 1)
	curNumBranches[1] = 2;
      else
	curNumBranches[i] = curNumBranches[i-1];

      // assign default parents for each branch:
      if (i > 0)
	assignDefaultBranchParents(curNumBranches[i-1], curNumBranches[i],
				   &(curBranchParents[i]), /*prevGen=*/i-1);
      // assign default of 0 samples to print
      curNumSampsToPrint[i] = new int[ curNumBranches[i] ];
      if (curNumSampsToPrint[i] == NULL) {
	printf("ERROR: out of memory");
	exit(5);
      }
      for (int b = 0; b < curNumBranches[i]; b++)
	curNumSampsToPrint[i][b] = 0;
    }

    int thisGenNumBranches;
    if (branchStr != NULL) {
      thisGenNumBranches = strtol(branchStr, &endptr, 10);
      if (errno != 0 || *endptr != '\0') {
	fprintf(stderr, "ERROR: line %d in def: optional third token must be numerical value giving\n",
		line);
	fprintf(stderr, "      number of branches\n");
	if (errno != 0)
	  perror("strtol");
	exit(2);
      }

      if (thisGenNumBranches <= 0) {
	fprintf(stderr, "ERROR: line %d in def: in generation %d, branch number zero or below\n",
		line, generation);
	exit(2);
      }
      else {
	curNumBranches[generation - 1] = thisGenNumBranches;
      }
    }
    else {
      if (generation - 1 == 0)
	thisGenNumBranches = 1;
      else if (generation - 1 == 1 && curNumBranches[0] == 1)
	thisGenNumBranches = 2;
      else
	thisGenNumBranches = curNumBranches[generation-2];

      curNumBranches[generation - 1] = thisGenNumBranches;
    }

    curNumSampsToPrint[generation - 1] = new int[thisGenNumBranches];
    if (curNumSampsToPrint[generation - 1] == NULL) {
      printf("ERROR: out of memory");
      exit(5);
    }
    for(int b = 0; b < thisGenNumBranches; b++) {
      curNumSampsToPrint[generation - 1][b] = numSamps;
    }

    lastReadGen = generation - 1;

    // now read in and assign (if only using the defaults) the branch parents
    // for this generation. Note that in the first generation, all individuals
    // are necessarily founders so there should not be any specification.
    if (generation - 1 > 0) {
      bool warning = readBranchSpec(curNumBranches,
			&curBranchParents[generation - 1],
			curNumSampsToPrint[generation - 1],
			/*curGen=*/generation - 1, curSexConstraints,
			/*prevSpouseNum=*/&curBranchNumSpouses[generation - 2],
			branchParentsAssigned, spouseDependencies, curI1Sex,
			delim, saveptr, endptr, line);
      warningGiven = warningGiven || warning;
    }
    else {
      bool warning = readBranchSpec(curNumBranches,
				    /*thisGenBranchParents=NA=*/NULL,
				    curNumSampsToPrint[generation - 1],
				    /*curGen=*/generation - 1,
				    curSexConstraints,
				    /*prevGenSpouseNum=NA=*/NULL,
				    branchParentsAssigned, spouseDependencies,
				    curI1Sex, delim, saveptr, endptr, line);
      warningGiven = warningGiven || warning;
    }
  }

  // do some bookkeeping to finalize the previously read pedigree
  int lastNumGen = curNumGen;
  finishLastDef(lastNumGen, curSexConstraints, spouseDependencies);

  for(auto it = simDetails.begin(); it != simDetails.end(); it++) {
    bool someBranchToPrint = false;
    bool anyNoPrint = false;
    int lastGenNumBranches = it->numBranches[ it->numGen - 1 ];
    for(int b = 0; b < lastGenNumBranches; b++) {
      if (it->numSampsToPrint[ it->numGen - 1 ][b] == 0)
	anyNoPrint = true;
      else
	someBranchToPrint = true;
    }

    if (!someBranchToPrint) {
      fprintf(stderr, "ERROR: request to simulate pedigree \"%s\" with %d generations\n",
	      it->name, it->numGen);
      fprintf(stderr, "       but no request to print any samples from last generation (number %d)\n",
	      it->numGen);
      exit(4);
    }
    else if (anyNoPrint) {
      fprintf(stderr, "Warning: no-print branches in last generation of pedigree %s:\n",
	      it->name);
      fprintf(stderr, "         can omit these branches and possibly reduce number of founders needed\n");
    }
  }

  if (simDetails.size() == 0) {
    fprintf(stderr, "ERROR: def file does not contain pedigree definitions;\n");
    fprintf(stderr, "       nothing to simulate\n");
    exit(3);
  }

  free(buffer);
  fclose(in);

  if (warningGiven)
    fprintf(stderr, "\n");
}

void finishLastDef(int numGen, SexConstraint **&sexConstraints,
		   vector< pair<set<Parent,ParentComp>,int8_t>* > &spouseDependencies) {
  // need to assign sex from spouse dependency sets to <sexConstraints>; also
  // don't need to store the spouse dependency sets anymore:
  for(unsigned int i = 0; i < spouseDependencies.size(); i++) {
    if (spouseDependencies[i] != NULL) {
      int8_t theSex = spouseDependencies[i]->second;
      if (theSex >= 0) {
	// assign the sexes of all samples in the spouseDependencies set to
	// <theSex>
	set<Parent,ParentComp> &toUpdate = spouseDependencies[i]->first;
	for(auto it = toUpdate.begin(); it != toUpdate.end(); it++) {
	  assert(sexConstraints[ it->gen ][ it->branch ].theSex == -1 ||
		 sexConstraints[ it->gen ][ it->branch ].theSex == theSex);
	  sexConstraints[ it->gen ][ it->branch ].theSex = theSex;
	}
      }
      delete spouseDependencies[i];
      spouseDependencies[i] = NULL;
    }
  }
  // TODO: can speed things by doing this, though note that it means that the
  // results for a given random seed will differ relative to an earlier version
  // of Ped-sim; therefore, will not do this, but may combine it with another
  // feature that has this effect
  // (Also, if this gets put in place, can remove the assignment of NULL just
  // above.)
  //spouseDependencies.clear();
}

// Gives the default parent assignment for any branches that have not had
// their parents explicitly specified.
void assignDefaultBranchParents(int prevGenNumBranches, int thisGenNumBranches,
				Parent **thisGenBranchParents, int prevGen,
				int *prevGenSpouseNum,
				vector<bool> *branchParentsAssigned) {
  // how many new branches is each previous branch the parent of?
  int multFactor = thisGenNumBranches / prevGenNumBranches;
  if (multFactor == 0)
    multFactor = 1; // for branches that survive, map prev branch i to cur i

  // allocate space to store the parents of each branch
  if (*thisGenBranchParents == NULL) {
    *thisGenBranchParents = new Parent[2 * thisGenNumBranches];
    if (*thisGenBranchParents == NULL) {
      printf("ERROR: out of memory");
      exit(5);
    }
  }

  for(int prevB = 0; prevB < prevGenNumBranches; prevB++) {
    if (prevB >= thisGenNumBranches)
      break; // defined all the branches for this generation
    // same founder spouse for all <multFactor> branches that <prevB>
    // is the parent of
    int spouseNum = -1;
    bool spouseNumDefined = false;

    for(int multB = 0; multB < multFactor; multB++) {
      int curBranch = prevB * multFactor + multB;
      if (branchParentsAssigned && (*branchParentsAssigned)[curBranch])
	continue; // skip assignment of branches that were assigned previously
      (*thisGenBranchParents)[curBranch*2].gen = prevGen;
      (*thisGenBranchParents)[curBranch*2].branch = prevB;
      if (!spouseNumDefined) {
	if (prevGenSpouseNum) {
	  prevGenSpouseNum[ prevB ]--;
	  spouseNum = prevGenSpouseNum[ prevB ];
	}
	else
	  spouseNum = -1;
	spouseNumDefined = true;
      }
      (*thisGenBranchParents)[curBranch*2 + 1].gen = prevGen;
      (*thisGenBranchParents)[curBranch*2 + 1].branch = spouseNum;
    }
  }
  // For any branches in this generation that are not an exact multiple of
  // the number of branches in the previous generation, make them brand new
  // founders. They will contain exactly one person regardless of the number
  // of samples to print
  for(int newB = prevGenNumBranches * multFactor; newB < thisGenNumBranches;
								      newB++) {
    if (branchParentsAssigned && (*branchParentsAssigned)[newB])
      continue; // skip assignment of branches that were assigned previously
    // undefined parents for excess branches: new founders
    (*thisGenBranchParents)[newB*2].gen = prevGen;
    (*thisGenBranchParents)[newB*2].branch =
				(*thisGenBranchParents)[newB*2 + 1].branch = -1;
  }
}

// Reads in and performs state changes for branch specifications including
// both parent assignments and no-print directives
// Returns true iff a warning has been printed
bool readBranchSpec(int *numBranches, Parent **thisGenBranchParents,
		    int *thisGenNumSampsToPrint, int curGen,
		    SexConstraint **sexConstraints, int **prevGenSpouseNum,
		    vector<bool> &branchParentsAssigned,
		    vector< pair<set<Parent,ParentComp>,int8_t>* > &spouseDependencies,
		    const int i1Sex, const char *delim, char *&saveptr,
		    char *&endptr, int line) {
  bool warningGiven = false;
  int prevGen = curGen - 1;

  assert(curGen == 0 || *prevGenSpouseNum == NULL);
  if (curGen > 0) {
    *prevGenSpouseNum = new int[numBranches[prevGen]];
    if (*prevGenSpouseNum == NULL) {
      printf("ERROR: out of memory\n");
      exit(5);
    }
    for(int b = 0; b < numBranches[prevGen]; b++)
      // What number have we assigned through for founder spouses of
      // individuals in the previous generation? Note that founders have an id
      // (in the code for the purposes of <curBranchParents>) that are always
      // negative and that by default we assign one founder spouse to marry one
      // person in each branch in the previous generation (see
      // assignDefaultBranchParents())
      (*prevGenSpouseNum)[b] = 0;

    *thisGenBranchParents = new Parent[ 2 * numBranches[curGen] ];
    if (*thisGenBranchParents == NULL) {
      printf("ERROR: out of memory\n");
      exit(5);
    }

    if (sexConstraints[prevGen] == NULL) {
      // if the previous generation didn't have any branch-specific sex
      // assignments, need to allocate space for sex constraints (i.e., based
      // on which branch i1 individual has children with which other branch i1)
      sexConstraints[prevGen] = new SexConstraint[numBranches[prevGen]];
      if (sexConstraints[prevGen] == NULL) {
	printf("ERROR: out of memory\n");
	exit(5);
      }
      initSexConstraints(sexConstraints[prevGen], numBranches[prevGen]);
    }

    // so far, all the branches in the current generation are assigned default
    // parents; track which branches get explicitly assigned and throw an error
    // if the same branch is assigned more than once
    branchParentsAssigned.clear();
    for(int i = 0; i < numBranches[curGen]; i++)
      branchParentsAssigned.push_back(false);
  }

  while (char *assignToken = strtok_r(NULL, delim, &saveptr)) {
    char *assignBranches = assignToken; // will add '\0' at ':'
    char *fullAssignPar = NULL;
    int i;
    Parent pars[2];

    // split on ':', 'n', or 's' to get the parent assignments/sexes on the
    // right and the branches on the left
    for(i = 0; assignToken[i] != ':' && assignToken[i] != 'n' &&
	       assignToken[i] != 's' && assignToken[i] != '\0'; i++);
    // ':' gives parents (after the ':'); n means the preceding branches
    // should not have their members printed; s means the following character
    // ('M' or 'F') gives the sex of the i1 individual in the branches
    bool parentAssign = false;
    bool noPrint = false;
    bool sexAssign = false;
    int sexToAssign = -1;
    if (assignToken[i] == ':')
      parentAssign = true;
    else if (assignToken[i] == 'n')
      noPrint = true;
    else if (assignToken[i] == 's')
      sexAssign = true;
    else {
      fprintf(stderr, "ERROR: line %d in def: improperly formatted parent assignment, sex assignment\n",
	      line);
      fprintf(stderr, "       or no-print field %s\n", assignToken);
      exit(8);
    }
    assignToken[i] = '\0';

    if (curGen == 0 && parentAssign) {
      fprintf(stderr, "ERROR: line %d in def: first generation cannot have parent specifications\n",
	      line);
      exit(8);
    }

    // should have only one of these options:
    assert(parentAssign || noPrint || sexAssign);

    if (parentAssign) {
      // Get the one or two parents
      char *assignPar[2];
      assignPar[0] = &(assignToken[i+1]); // will add '\0' at '_' if present
      assignPar[1] = NULL; // initially; updated just below

      readParents(numBranches, prevGen, sexConstraints, prevGenSpouseNum,
		  spouseDependencies, assignBranches, assignPar, pars,
		  fullAssignPar, i1Sex, endptr, line);
    }
    else if (noPrint) {
      // expect a space after the 'n': check this
      if (assignToken[i+1] != '\0') {
	assignToken[i] = 'n';
	fprintf(stderr, "ERROR: line %d in def: improperly formatted no-print field \"%s\":\n",
		line, assignToken);
	fprintf(stderr, "       no-print character 'n' should be followed by white space\n");
	exit(8);
      }
    }
    else {
      bool badField = false;
      if (assignToken[i+1] == 'M')
	sexToAssign = 0;
      else if (assignToken[i+1] == 'F')
	sexToAssign = 1;
      else
	badField = true;

      if (badField || assignToken[i+2] != '\0') {
	assignToken[i] = 's';
	fprintf(stderr, "ERROR: line %d in def: improperly formatted sex assignment field \"%s\":\n",
		line, assignToken);
	fprintf(stderr, "       character 's' should be followed either 'M' or 'F' and then white space\n");
	exit(10);
      }
    }

    // either we're not assigning a sex or <sexToAssign> is 0 or 1:
    assert(!sexAssign || sexToAssign == 0 || sexToAssign == 1);

    // process the branches
    // if <parentAssign>, these will be assigned <pars> as parent OR
    // if <noPrint>, these will not be printed OR
    // if <sexAssign>, these will be assigned <sexToAssign> as their sex
    bool done = false;
    // the starting branch for a range (delimited by '-'); see below
    char *startBranch = NULL;
    while (!done) {
      for(i = 0; assignBranches[i] != ',' && assignBranches[i] != '-' &&
		 assignBranches[i] != '\0'; i++);
      if (assignBranches[i] == '-') { // have a range; just passed over start:
	assignBranches[i] = '\0';
	if (startBranch != NULL) {
	  fprintf(stderr, "ERROR: line %d in def: improperly formatted branch range \"%s-%s-\"\n",
		  line, startBranch, assignBranches);
	  exit(5);
	}
	startBranch = assignBranches;
	assignBranches = &(assignBranches[i+1]); // go through next loop
      }
      else if (assignBranches[i] == ',' || assignBranches[i] == '\0') {
	if (assignBranches[i] == '\0') {
	  done = true;
	  if (i == 0)
	    break; // early termination, will get error below
	}
	else
	  assignBranches[i] = '\0';

	int curBranch = strtol(assignBranches, &endptr, 10) - 1; // 0 indexed
	if (errno != 0 || *endptr != '\0') {
	  fprintf(stderr, "ERROR: line %d in def: unable to parse branch %s to ",
		  line, assignBranches);
	  if (parentAssign)
	    fprintf(stderr, "assign parent %s to\n", fullAssignPar);
	  else if (noPrint)
	    fprintf(stderr, "set as no-print\n");
	  else // sexAssign
	    fprintf(stderr, "assign sex %c to\n",
		    (sexToAssign == 0) ? 'M' : 'F');
	  if (errno != 0)
	    perror("strtol");
	  exit(2);
	}

	if (startBranch) {
	  int rangeEnd = curBranch;
	  int rangeStart = strtol(startBranch, &endptr, 10) - 1; // 0 indexed
	  if (errno != 0 || *endptr != '\0') {
	    fprintf(stderr, "ERROR: line %d in def: unable to parse branch %s to ",
		    line, startBranch);
	    if (parentAssign)
	      fprintf(stderr, "assign parent %s to\n", fullAssignPar);
	    else if (noPrint)
	      fprintf(stderr, "set as no-print\n");
	    else // sexAssign
	      fprintf(stderr, "assign sex %c to\n",
		      (sexToAssign == 0) ? 'M' : 'F');
	    if (errno != 0)
	      perror("strtol");
	    exit(2);
	  }
	  startBranch = NULL; // parsed: reset this variable

	  if (rangeStart >= rangeEnd) {
	    fprintf(stderr, "ERROR: line %d in def: non-increasing branch range %d-%d to\n",
		    line, rangeStart, rangeEnd);
	    if (parentAssign)
	      fprintf(stderr, "       assign parent %s to\n", fullAssignPar);
	    else if (noPrint)
	      fprintf(stderr, "       set as no-print\n");
	    else // sexAssign
	      fprintf(stderr, "       assign sex %c to\n",
		      (sexToAssign == 0) ? 'M' : 'F');
	    exit(8);
	  }
	  if (rangeEnd >= numBranches[curGen]) {
	    fprintf(stderr, "ERROR: line %d in def: request to assign a branch greater than %d, the total\n",
		    line, numBranches[curGen]);
	    fprintf(stderr, "       number of branches in generation %d\n", curGen + 1);
	    exit(11);
	  }

	  for(int branch = rangeStart; branch <= rangeEnd; branch++) {
	    assignBranch(parentAssign, noPrint, sexToAssign, curGen, branch,
			 sexConstraints, thisGenBranchParents,
			 thisGenNumSampsToPrint, numBranches[curGen],
			 branchParentsAssigned, pars, line, warningGiven);
	  }
	}
	else {
	  if (curBranch >= numBranches[curGen]) {
	    fprintf(stderr, "ERROR: line %d in def: request to assign a branch greater than %d, the total\n",
		    line, numBranches[curGen]);
	    fprintf(stderr, "       number of branches in generation %d\n", curGen + 1);
	    exit(11);
	  }

	  assignBranch(parentAssign, noPrint, sexToAssign, curGen, curBranch,
		       sexConstraints, thisGenBranchParents,
		       thisGenNumSampsToPrint, numBranches[curGen],
		       branchParentsAssigned, pars, line, warningGiven);
	}

	assignBranches = &(assignBranches[i+1]); // go through next in loop
      }
    }

    if (startBranch != NULL) {
      fprintf(stderr, "ERROR: line %d in def: range of branches ", line);
      if (parentAssign)
	fprintf(stderr, "to assign parents ");
      else if (noPrint)
	fprintf(stderr, "set as no-print ");
      else // sexAssign
	fprintf(stderr, "assign sex %c to\n", (sexToAssign == 0) ? 'M' : 'F');
      fprintf(stderr, "does not terminate\n");
      exit(8);
    }
  }

  // TODO: As a space optimization, could compress the spouseDependencies
  //       vector more. When combining two sets, the indexes of one of the set
  //       are no longer used, but the size of the spouseDependencies list
  //       still accounts for them.
  assert(spouseDependencies.size() % 2 == 0);

  if (curGen > 0)
    assignDefaultBranchParents(numBranches[prevGen], numBranches[curGen],
			       thisGenBranchParents, prevGen, *prevGenSpouseNum,
			       &branchParentsAssigned);

  return warningGiven;
}

void assignBranch(bool parentAssign, bool noPrint, int sexToAssign, int curGen,
		  int branch, SexConstraint **sexConstraints,
		  Parent **thisGenBranchParents, int *thisGenNumSampsToPrint,
		  int thisGenNumBranches, vector<bool> &branchParentsAssigned,
		  Parent pars[2], int line, bool &warningGiven) {
  if (parentAssign) {
    if (branchParentsAssigned[branch]) {
      fprintf(stderr, "ERROR: line %d in def: parents of branch number %d assigned multiple times\n",
	      line, branch+1);
      exit(8);
    }
    branchParentsAssigned[branch] = true;
    for(int p = 0; p < 2; p++)
      (*thisGenBranchParents)[branch*2 + p] = pars[p];
  }
  else if (noPrint) {
    // print 0 samples for <branch>
    if (thisGenNumSampsToPrint[branch] > 1) {
      fprintf(stderr, "Warning: line %d in def: generation %d would print %d individuals, now set to 0\n",
	      line, curGen + 1, thisGenNumSampsToPrint[branch]);
      warningGiven = true;
    }
    else if (thisGenNumSampsToPrint[branch] == 0) {
      fprintf(stderr, "Warning: line %d in def: generation %d branch %d, no-print is redundant\n",
	      line, curGen + 1, branch + 1);
      warningGiven = true;
    }
    thisGenNumSampsToPrint[branch] = 0;
  }
  else { // sexAssign
    if (sexConstraints[curGen] == NULL) {
      // make space for sex assignments
      sexConstraints[curGen] = new SexConstraint[thisGenNumBranches];
      if (sexConstraints[curGen] == NULL) {
	printf("ERROR: out of memory\n");
	exit(5);
      }
      initSexConstraints(sexConstraints[curGen], thisGenNumBranches);
    }
    else if (sexConstraints[curGen][branch].theSex != -1) {
      fprintf(stderr, "ERROR: line %d in def: sex of branch number %d assigned multiple times\n",
	      line, branch+1);
      exit(8);
    }
    sexConstraints[curGen][branch].theSex = sexToAssign;
  }
}

// In the branch specifications, read the parent assignments
void readParents(int *numBranches, int prevGen, SexConstraint **sexConstraints,
		 int **prevGenSpouseNum,
		 vector< pair<set<Parent,ParentComp>,int8_t>* > &spouseDependencies,
		 char *assignBranches, char *assignPar[2], Parent pars[2],
		 char *&fullAssignPar, const int i1Sex, char *&endptr,
		 int line) {
  int i;

  errno = 0; // initially

  // Find the second parent if present
  for(i = 0; assignPar[0][i] != '_' && assignPar[0][i] != '\0'; i++);
  if (assignPar[0][i] == '_') {
    assignPar[0][i] = '\0';
    assignPar[1] = &(assignPar[0][i+1]);
  }

  for(int p = 0; p < 2; p++) {
    pars[p].branch = -1;
    pars[p].gen = prevGen;
  }
  for(int p = 0; p < 2 && assignPar[p] != NULL && assignPar[p][0] != '\0'; p++){
    // Check for generation number
    char *genNumStr = NULL;
    for(i = 0; assignPar[p][i] != '^' && assignPar[p][i] != '\0'; i++);
    if (assignPar[p][i] == '^') {
      // Have a generation number
      if (p == 0) {
	fprintf(stderr, "ERROR: line %d in def: parent assignment for branches %s gives generation\n",
		line, assignBranches);
	fprintf(stderr, "       number for the first parent, but this is only allowed for the second\n");
	fprintf(stderr, "       parent; for example, 2:1_3^1 has branch 1 from previous generation\n");
	fprintf(stderr, "       married to branch 3 from generation 1\n");
	exit(3);
      }
      assignPar[p][i] = '\0';
      genNumStr = &(assignPar[p][i+1]);
      pars[p].gen = strtol(genNumStr, &endptr, 10) - 1; // 0 indexed => -1
      if (errno != 0 || *endptr != '\0') {
	fprintf(stderr, "ERROR: line %d in def: unable to parse parent assignment for branches %s\n",
		line, assignBranches);
	fprintf(stderr, "       malformed generation number string for second parent: %s\n",
		genNumStr);
	if (errno != 0)
	  perror("strtol");
	exit(5);
      }
      if (pars[p].gen > prevGen) {
	fprintf(stderr, "ERROR: line %d in def: unable to parse parent assignment for branches %s\n",
		line, assignBranches);
	fprintf(stderr, "       generation number %s for second parent is after previous generation\n",
		genNumStr);
	exit(-7);
      }
      else if (pars[p].gen < 0) {
	fprintf(stderr, "ERROR: line %d in def: unable to parse parent assignment for branches %s\n",
		line, assignBranches);
	fprintf(stderr, "       generation number %s for second parent is before first generation\n",
		genNumStr);
	exit(5);
      }
    }

    pars[p].branch = strtol(assignPar[p], &endptr, 10) - 1; // 0 indexed => -1
    if (errno != 0 || *endptr != '\0') {
      fprintf(stderr, "ERROR: line %d in def: unable to parse parent assignment for branches %s\n",
	      line, assignBranches);
      if (errno != 0)
	perror("strtol");
      exit(2);
    }
    if (pars[p].branch < 0) {
      fprintf(stderr, "ERROR: line %d in def: parent assignments must be of positive branch numbers\n",
	      line);
      exit(8);
    }
    else if (pars[p].branch >= numBranches[ pars[p].gen ]) {
      fprintf(stderr, "ERROR: line %d in def: parent branch number %d is more than the number of\n",
	      line, pars[p].branch+1);
      fprintf(stderr, "       branches (%d) in generation %d\n",
	      numBranches[ pars[p].gen ], pars[p].gen+1);
      exit(8);
    }
    // so that we can print the parent assignment in case of errors below
    if (genNumStr != NULL)
      genNumStr[-1] = '^';
  }
  if (pars[0].branch == -1) {
    // new founder
    assert(pars[1].branch == -1);
  }
  else if (pars[1].branch == -1) {
    // Have not yet assigned the numerical id of parent 1. Because the def
    // file doesn't specify this, it is a founder, and one that hasn't been
    // assigned before. As such, we'll get a unique number associated with a
    // spouse of pars[0].branch. Negative values correspond to founders, so
    // we decrement <prevGenSpouseNum>. (It is initialized to 0 above)
    (*prevGenSpouseNum)[ pars[0].branch ]--;
    pars[1].branch = (*prevGenSpouseNum)[ pars[0].branch ];
  }
  else {
    if (pars[0].branch == pars[1].branch && pars[0].gen == pars[1].gen) {
      fprintf(stderr, "ERROR: line %d in def: cannot have both parents be from same branch\n",
	      line);
      exit(8);
    }
    if (i1Sex >= 0) {
      fprintf(stderr, "ERROR: line %d in def: cannot have fixed sex for i1 samples and marriages\n",
	      line);
      fprintf(stderr, "       between branches -- i1's will have the same sex and cannot reproduce.\n");
      fprintf(stderr, "       consider assigning sexes to individual branches\n");
      exit(9);
    }
    updateSexConstraints(sexConstraints, pars, numBranches, spouseDependencies,
			 line);
  }

  // so that we can print the parent assignment in case of errors below
  if (assignPar[1] != NULL)
    assignPar[1][-1] = '_';
  fullAssignPar = assignPar[0];
}

// Given the branch indexes of two parents, adds constraints and error checks
// to ensure that this couple does not violate the requirement that parents
// must have opposite sex.
// Also stores (in <spouseDependencies>) the specified or inferred sexes of the
// branches; sexes explicitly specified in the def file are currently stored in
// <sexConstraints>. This function infers the sexes of those branches these
// individuals have children with, and transitively to the partners of those
// branches, etc.
void updateSexConstraints(SexConstraint **sexConstraints, Parent pars[2],
			  int *numBranches,
			  vector< pair< set<Parent,ParentComp>, int8_t>* >
							    &spouseDependencies,
			  int line) {
  // we check these things in the caller, but just to be sure:
  for(int p = 0; p < 2; p++) {
    assert(pars[p].branch >= 0);
    assert(pars[p].branch < numBranches[ pars[p].gen ]);
  }

  if (sexConstraints[ pars[0].gen ][ pars[0].branch ].set == -1 &&
      sexConstraints[ pars[1].gen ][ pars[1].branch ].set == -1) {
    // neither is a member of a spouse set: create and add to
    // <spouseDependencies>
    pair<set<Parent,ParentComp>,int8_t> *sets[2];
    for(int p = 0; p < 2; p++) {
      sets[p] = new pair<set<Parent,ParentComp>,int8_t>();
      if (sets[p] == NULL) {
	printf("ERROR: out of memory");
	exit(5);
      }
      sets[p]->first.insert( pars[p] );
      // which index in spouseDependencies is the set corresponding to this
      // parent stored in? As we're about to add it just below this, the
      // current size will be the index
      sexConstraints[ pars[p].gen ][ pars[p].branch ].set =
						      spouseDependencies.size();
      if (sexConstraints[ pars[p].gen ][ pars[p].branch ].theSex != -1)
	sets[p]->second = sexConstraints[pars[p].gen][pars[p].branch].theSex;
      else
	sets[p]->second = -1;
      spouseDependencies.push_back( sets[p] );
    }

    // any assigned sexes? ensure consistency:
    if (sets[0]->second >= 0 || sets[1]->second >= 0) {
      for(int p = 0; p < 2; p++)
	if (sets[p]->second < 0)
	  // infer sex that is unassigned: is opposite (^ 1) the assigned one
	  sets[p]->second = sets[ p^1 ]->second ^ 1;
      if (sets[0]->second != (sets[1]->second ^ 1)) {
	fprintf(stderr, "ERROR: line %d in def: assigning branch %d from generation %d and branch %d from\n",
		line, pars[0].branch+1, pars[0].gen+1, pars[1].branch+1);
	fprintf(stderr, "       generation %d as parents is impossible: they are assigned the same sex\n",
		pars[1].gen+1);
	exit(3);
      }
    }
  }
  else if (sexConstraints[ pars[0].gen ][ pars[0].branch ].set == -1 ||
	   sexConstraints[ pars[1].gen ][ pars[1].branch ].set == -1) {
    // one spouse is a member of a spouse set and the other is not: add the
    // other parent to the opposite spouse set, error check, and update state
    int assignedPar = -1;
    if (sexConstraints[ pars[0].gen ][ pars[0].branch ].set >= 0)
      assignedPar = 0;
    else
      assignedPar = 1;

    int otherPar = assignedPar ^ 1;
    int assignedSetIdx = sexConstraints[ pars[assignedPar].gen ]
				       [ pars[assignedPar].branch ].set;

    // since <otherPar> isn't in a spouse set yet, it definitely shouldn't be in
    // the same spouse set as <assignedPar>
    assert((unsigned int) assignedSetIdx < spouseDependencies.size());
    assert(spouseDependencies[ assignedSetIdx ] != NULL);
    assert(spouseDependencies[ assignedSetIdx ]->first.find( pars[otherPar] ) ==
			      spouseDependencies[assignedSetIdx]->first.end());

    // NOTE: The following is a trick that relies on the fact that sets are
    // stored as sequential pairs in <spouseDependencies>. Ex: indexes 0 and 1
    // are pairs of spouses that are dependent upon each other (those in index
    // 0 must have the same sex and must be opposite those in index 1). To get
    // an even number that is 1 minus an odd value or the odd number that is 1
    // plus an even value, it suffices to flip the lowest order bit:
    int otherSetIdx = assignedSetIdx ^ 1;
    // add <pars[otherPar]> to the set containing spouses of <pars[assignedPar]>
    spouseDependencies[ otherSetIdx ]->first.insert( pars[otherPar] );
    sexConstraints[ pars[otherPar].gen ][ pars[otherPar].branch ].set
								  = otherSetIdx;
    // is the "new" spouse assigned a sex? If not, no change needed: that
    // person can have sex assigned according to the current constraints, but
    // if so, error check and, if needed, update
    if (sexConstraints[pars[otherPar].gen][pars[otherPar].branch].theSex >= 0) {
      if (spouseDependencies[ otherSetIdx ]->second == -1) {
	// sex not yet assigned in <spouseDependencies>; ensure that that's
	// true of both the "linked" dependency sets:
	assert(spouseDependencies[ assignedSetIdx ]->second == -1);

	// no sexes assigned yet; assign:
	spouseDependencies[ otherSetIdx ]->second =
	       sexConstraints[pars[otherPar].gen][pars[otherPar].branch].theSex;
	spouseDependencies[ assignedSetIdx ]->second =
				(spouseDependencies[ otherSetIdx ]->second ^ 1);
      }
      else if (spouseDependencies[ otherSetIdx ]->second !=
	     sexConstraints[pars[otherPar].gen][pars[otherPar].branch].theSex) {
	fprintf(stderr, "ERROR: line %d in def: assigning branch %d from generation %d as a parent with\n",
		line, pars[otherPar].branch+1, pars[otherPar].gen+1);
	fprintf(stderr, "       branch %d from generation %d is impossible: due to sex assignments and/or\n",
		pars[assignedPar].branch+1, pars[assignedPar].gen+1);
	fprintf(stderr, "       other parent assignments they necessarily have the same sex\n");
	exit(4);
      }
      // else: the current assigned sex for <otherSetIdx> is the same as the
      // spouse
    }
  }
  else {
    assert(sexConstraints[ pars[0].gen ][ pars[0].branch ].set >= 0 &&
	   sexConstraints[ pars[1].gen ][ pars[1].branch ].set >= 0);
    // both are members of spouse sets

    if (sexConstraints[ pars[0].gen ][ pars[0].branch ].set / 2 ==
	sexConstraints[ pars[1].gen ][ pars[1].branch ].set / 2) {
      // spouse sets from the same pair of sets
      if (sexConstraints[ pars[0].gen ][ pars[0].branch ].set ==
	  sexConstraints[ pars[1].gen ][ pars[1].branch ].set) {
	fprintf(stderr, "ERROR: line %d in def: assigning branch %d from generation %d and branch %d from\n",
		line, pars[0].branch+1, pars[0].gen+1, pars[1].branch+1);
	fprintf(stderr, "       generation %d as parents is impossible due to other parent assignments:\n",
		pars[1].gen+1);
	fprintf(stderr, "       they necessarily have same sex\n");
	exit(5);
      }
      // otherwise done with set assignment
    }
    else {
      // two distinct sets of spouse sets: must generate two sets that are the
      // unions of the appropriate sets -- all the spouses of both must be
      // constrained to be opposite sex
      int setIdxes[2][2]; // current set indexes
      for(int p = 0; p < 2; p++) {
	setIdxes[p][0] = sexConstraints[ pars[p].gen ]
				       [ pars[p].branch ].set;
	// see comment denoted with NOTE a little ways above for why this works:
	setIdxes[p][1] = setIdxes[p][0] ^ 1;
      }

      for(int i = 0; i < 2; i++)
	for(int j = 0; j < 2; j++)
	  assert(spouseDependencies[ setIdxes[i][j] ] != NULL);

      // check/ensure that the sets to be unioned have consistent sexes (if
      // assigned). We will union into setIdxes[0][*], so loop over the second
      // index (the merge parent index)
      for(int mp = 0; mp < 2; mp++) { // loop over merged p index
	int op = mp ^ 1; // other p
	if (spouseDependencies[ setIdxes[0][mp] ]->second !=
	    spouseDependencies[ setIdxes[1][op] ]->second) {
	  if (spouseDependencies[ setIdxes[0][mp] ]->second == -1)
	    // set to be unioned into doesn't have a sex assigned; set to the
	    // same as the other parent:
	    spouseDependencies[ setIdxes[0][mp] ]->second =
				  spouseDependencies[ setIdxes[1][op] ]->second;
	  else if (spouseDependencies[ setIdxes[1][op] ]->second != -1) {
	    // the sexes of the two sets to be merged are different; error
	    fprintf(stderr, "ERROR: line %d in def: assigning branch %d from generation %d as a parent with\n",
		    line, pars[0].branch+1, pars[0].gen+1);
	    fprintf(stderr, "       branch %d from generation %d is impossible: due to sex assignments and/or\n",
		    pars[1].branch+1, pars[1].gen+1);
	    fprintf(stderr, "       other parent assignments they necessarily have the same sex\n");
	    exit(6);
	  }
	}
      }
      // sanity check that the end result is either opposite sex assignments or
      // no sex assignments
      if (spouseDependencies[ setIdxes[0][0] ]->second != -1)
	assert(spouseDependencies[ setIdxes[0][0] ]->second == 
			    (spouseDependencies[ setIdxes[0][1] ]->second ^ 1));
      else {
	assert(spouseDependencies[ setIdxes[0][1] ]->second == -1);
	assert(spouseDependencies[ setIdxes[1][0] ]->second == -1);
	assert(spouseDependencies[ setIdxes[1][1] ]->second == -1);
      }

      // This is legacy code, but I'll keep it anyway:
      // ensure the values are different pointers (old code used to update
      // the pointers directly instead of setting one of the two merged sets
      // to NULL)
      assert(spouseDependencies[ setIdxes[0][0] ] !=
					spouseDependencies[ setIdxes[1][1] ]);
      assert(spouseDependencies[ setIdxes[0][1] ] !=
					  spouseDependencies[ setIdxes[1][0] ]);

      // take the union such that spouses of each are contained in both
      spouseDependencies[ setIdxes[0][0] ]->first.insert(
			    spouseDependencies[ setIdxes[1][1] ]->first.begin(),
			    spouseDependencies[ setIdxes[1][1] ]->first.end());
      spouseDependencies[ setIdxes[0][1] ]->first.insert(
			    spouseDependencies[ setIdxes[1][0] ]->first.begin(),
			    spouseDependencies[ setIdxes[1][0] ]->first.end());

      // intersection of these should be empty, otherwise there's
      // inconsistencies and individuals that are meant to be different sexes
      // simultaneously
      // seems like this is impossible, so arguably should be an assertion, but
      // we'll keep the informative error in:
      if (intersectNonEmpty(spouseDependencies[ setIdxes[0][0] ]->first,
			    spouseDependencies[ setIdxes[0][1] ]->first)) {
	fprintf(stderr, "ERROR: line %d in def: assigning branch %d from generation %d and branch %d from\n",
		line, pars[0].branch+1, pars[0].gen+1, pars[1].branch+1);
	fprintf(stderr, "       generation %d as parents is impossible due to other parent assignments:\n",
		pars[1].gen+1);
	fprintf(stderr, "       they necessarily have same sex\n");
	exit(7);
      }

      // replace set values stored in sexConstraints to the newly merged set
      // indexes
      for(int p = 0; p < 2; p++) {
	// update all set indexes for samples in
	// <spouseDependencies[ setIdxes[1][p] ]> to setIdxes[0][ p^1 ];
	set<Parent,ParentComp> &toUpdate =
				    spouseDependencies[ setIdxes[1][p] ]->first;
	for(auto it = toUpdate.begin(); it != toUpdate.end(); it++)
	  sexConstraints[ it->gen ][ it->branch ].set = setIdxes[0][p^1];

	delete spouseDependencies[ setIdxes[1][p] ];
	spouseDependencies[ setIdxes[1][p] ] = NULL;
      }
    }
  }

  // the sexes of these branches (if assigned) should now be set in
  // <spouseDependencies>
  for(int p = 0; p < 2; p++) {
    int assignedSetIdx = sexConstraints[ pars[p].gen ][ pars[p].branch ].set;
    assert(sexConstraints[ pars[p].gen ][ pars[p].branch ].theSex == -1 ||
	   spouseDependencies[ assignedSetIdx ]->second ==
			sexConstraints[ pars[p].gen ][ pars[p].branch ].theSex);
  }
}

// Returns true if the intersection of <a> and <b> is non-empty
bool intersectNonEmpty(set<Parent,ParentComp> &a, set<Parent,ParentComp> &b) {
  // Similar to the std::set_intersection code
  auto a_it = a.begin();
  auto b_it = b.begin();
  while(a_it != a.end() && b_it != b.end()) {
    if (ParentComp()(*a_it, *b_it))
      a_it++;
    else if (ParentComp()(*b_it, *a_it))
      b_it++;
    else {
      // have *a_it == *b_it, so intersection is non-empty:
      return true;
    }
  }
  return false;
}

void initSexConstraints(SexConstraint *newSexConstraints, int numBranches) {
  for(int i = 0; i < numBranches; i++) {
    newSexConstraints[i].set = -1;
    newSexConstraints[i].theSex = -1;
  }
}

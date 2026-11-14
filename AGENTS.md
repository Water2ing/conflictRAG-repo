```markdown
Always do:
- Agent has to always review and question the robustness of the methodology for a paper that will be submitted to an A/A* venue (NAACL)
- Make sure that the results after each run is stored into log file (log.jsonl)
- Whenever make a change, write down a description and note in CHANGE_LOG file. Make a copy of the files before making any changes

Ask first:
- Any change to the principles/core methodologies of the paper
- How to structure/store/name a file, naming convention needs to be clear/consistent

Never do:
- Run GPU-heavy tasks with heavy local model on the local machine/laptop




File structure

/Khang-Vu-EMNLP-2026-conflictrag

- Store the reference/rough draft for the paper
- This is the suggestion/idea, not the experimental/empirical data for the paper
- DO NOT use any data from this directory for the experiment or actual paper

/Khang-Vu-EMNLP-2026-conflictrag/Khang-Vu-EMNLP-2026-conflictrag        # The draft given by my professor, with purely speculative and theoretical data


/conflictRAG-repo
- Where the actual executed scripts, results, and data are stored
- This is also a local repo for a github repo named "conflictRAG-repo" that is used to share the scripts and results with my peers"

root directory at /conflictRAG
- Where dependencies and auxillary files are stored
- File downloaded from the server is also downloaded and stored here before being moved to /conflictRAG-repo




Note on workflow:
- The server and local repo has slightly different file structure
- View server_connect.txt for more details on commonly used commands when running baselines/tests on the server
- server_connect.txt has command for 2 different server, but the initialization is the same
- There are many redundant/old files that hasnt been updated or delete. Consult user whenver confussed
- There are only 2 development files that need to be focus on: "joint_grid_search.py" and "train_policy.py"
- When running test on the server, if port 8080 is not available, change to 8081


```

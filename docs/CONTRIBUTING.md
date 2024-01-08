# Contributing to PGT

First of all, thank you for taking time to make contributions to PGT!
This file provides the more technical guidelines on how to realize it.
For more non-technical aspects, please refer to the [PGT Contribution Guide](./community/contribution_guide.md)

## Table of Contents

- [Got a question?](#got-a-question)
- [Structure of the package](#structure-of-the-package)
- [Submitting an Issue](#submitting-an-issue)
- [Submitting a Pull Request](#submitting-a-pull-request)
- [Commit message guidelines](#commit-message-guidelines)

## Got a question?

Please referring to our GitHub [issue tracker](https://github.com/RUC-MIALAB/PGT/issues), and our developers are willing to help.
If you find a bug, you can help us by submitting an issue to our GitHub Repository. Even better, you can submit a Pull Request with a patch. You can request a new feature by submitting an issue to our GitHub Repository.
If you would like to implement a new feature, please submit an issue with a proposal for your work first, and that ensures your work collaborates with our development road map well. For a major feature, first open an issue and outline your proposal so that it can be discussed. This will also allow us to better coordinate our efforts, prevent duplication of work, and help you to craft the change so that it is successfully accepted into the project.

## Structure of the package

Please refer to [our instructions](./quick_start/easy_install.md) on how to installing PGT.
The source code of PSAIA is based on several modules. Under the PGT root directory, there are the following folders:

- `docs`: documents and supplementary info about PGT;
- `example`: some examples showing the usage of PGT;
- esm: files of ESM pre-training model;
- model: model files of PGT;
- train: training files of PGT;
- weight：training weight file of PGT;
- util: other files of PGT;

For those who are interested in the source code, the following figure shows the structure of the source code.

```
|-- predict                        A basic file including 
|-- util_feature                   extracte protein features
|-- model                          Run PGT model, predict inter-chain contact map
|--                                calculate PGT scores of residue pairs, give a ranking of residue pairs,take the top residue pairs 
```

## Submitting an Issue

Before you submit an issue, please search the issue tracker, and maybe your problem has been discussed and fixed. You can [submit new issues]((https://github.com/RUC-MIALAB/PGT/issues)/new/choose) by filling our issue forms.
To help us reproduce and confirm a bug, please provide a test case and building environment in your issue.

## Submitting a Pull Request

1. [Fork](https://docs.github.com/en/github/getting-started-with-github/fork-a-repo) the [PSAIA repository](https://github.com/RUC-MIALAB/PGT). If you already had an existing fork, [sync](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork) the fork to keep your modification up-to-date.

2. Pull your forked repository, create a new git branch, and make your changes in it:

     ```shell
     git checkout -b my-fix-branch
     ```

3. Coding your patch, including appropriate test cases and docs.
To run a subset of unit test, use `ctest -R <test-match-pattern>` to perform tests with name matched by given pattern.

4. After tests passed, commit your changes [with a proper message](#commit-message-guidelines).

5. Push your branch to GitHub:

    ```shell
    git push origin my-fix-branch
    ```

6. In GitHub, send a pull request (PR) with `RUC-MIALAB/PGT:PGT` as the base repository. It is required to document your PR following [our guidelines](#commit-message-guidelines).

7. After your pull request is merged, you can safely delete your branch and sync the changes from the main (upstream) repository:

- Delete the remote branch on GitHub either [through the GitHub web UI](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-branches-in-your-repository/deleting-and-restoring-branches-in-a-pull-request#deleting-a-branch-used-for-a-pull-request) or your local shell as follows:

    ```shell
    git push origin --delete my-fix-branch
    ```

- Check out the master branch:

    ```shell
    git checkout develop -f
    ```

- Delete the local branch:

    ```shell
    git branch -D my-fix-branch
    ```

- Update your master with the latest upstream version:

    ```shell
    git pull --ff upstream develop
    ```

## Commit message guidelines

A well-formatted commit message leads a more readable history when we look through some changes, and helps us generate change log.
We follow up [The Conventional Commits specification](https://www.conventionalcommits.org) for commit message format.
This format is also required for PR title and message.
The commit message should be structured as follows:

```text
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

- Header
  - type: The general intention of this commit
    - `Feature`: A new feature
    - `Fix`: A bug fix
    - `Docs`: Only documentation changes
    - `Style`: Changes that do not affect the meaning of the code
    - `Refactor`: A code change that neither fixes a bug nor adds a feature
    - `Perf`: A code change that improves performance
    - `Test`: Adding missing tests or correcting existing tests
    - `Build`: Changes that affect the build system or external dependencies
    - `CI`: Changes to our CI configuration files and scripts
    - `Revert`: Reverting commits
  - scope: optional, could be the module which this commit changes; for example, `orbital`
  - description: A short summary of the code changes: tell others what you did in one sentence.
- Body: optional, providing detailed, additional, or contextual information about the code changes, e.g. the motivation of this commit, referenced materials, the coding implementation, and so on.
- Footer: optional, reference GitHub issues or PRs that this commit closes or is related to. [Use a keyword](https://docs.github.com/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue#linking-a-pull-request-to-an-issue-using-a-keyword) to close an issue, e.g. "Fix #753".

Here is an example:

```text
Fix(sort_patches): use correct scores to sort.

`pzgemv_` and `pzgemm_` used `double*` for alpha and beta parameters but not `complex*` , this would cause error in GNU compiler.

Fix #753.
```

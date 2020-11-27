# first_project

To start the notebook server:

```
$ cargo make
```
or (without [cargo](https://doc.rust-lang.org/cargo/getting-started/installation.html) and [cargo-make](https://sagiegurari.github.io/cargo-make/#installation))
```
$ jupyter notebook --browser firefox Project1.ipynb
```

## Git tutorial

[How to set up git on your machine](https://kbroman.org/github_tutorial/pages/first_time.html)

### How to clone a git repository
In a terminal, move to a directory where you want to download this repository to, for example: 
```
$ cd ~/git
```

Then "clone" this repository to your local machine:
```
$ git clone https://github.com/mankinskin/first_project.git
```
The repository will be downloaded and put in a directory at `~/git/first_project`.

`cd` into the repository.

### Updating the local repository with the remote repository

*(this is not necessary after you just cloned the repository, but should be done regularly)*

You now have a copy of the "remote" git repository (stored on github) on your local machine.
When other contributors make changes to the repo, you need to update your local version by pulling in the latest commit history:
```
$ git pull
```
#### Pull with rebase
A common issue is that you forgot to `pull` before starting your work, and already made some commits to your local repository.
In this case you need to "rebase" your local repository on the remote repository.
That means git will first pull any new commits from remote and then apply your new local commits afterwards:
```
$ git pull -r
```
(you can make pull rebase by default using this: `git config --global pull.rebase true`)

### Pushing changes to the remote repository

To publish changes to the remote repository, you need to perform these steps:
- Make changes to tracked files or create new files (not tracked yet)
- Add changes or files to the staging area (prepare them to be committed)
- Make a commit (a package of changes with a description)
- Push your local commits to the remote repository

#### Adding changes and files to the staging area

You can add things to the "staging area" to commit them in the next step. Some files are already tracked by git.
You can get an overview with `git status`. It will show you untracked items, tracked items with changes and staged changes.

To add an entire file or directory to be commited, use:
```
$ git add my_file
```
To add all changes to tracked files, use `add --update`:
```
$ git add -u
```
To selectively pick changes to be added use `--patch`:
```
$ git add -u -p
```
[There are a lot more things you could do](https://git-scm.com/docs/git-add)

#### Creating a new commit

You now have changes in the staging area. Now you can put them in a commit which can be pushed to the remote repository.
You should provide a good description of what the commit *does* to the repository. I would advise you to use this command:
```
$ git commit -v
```
this should open an editor in your terminal where you can review the changes and can edit the commit message on multiple lines.
A short version would be
```
$ git commit -m "Add solution for kNN-Classifier"
```
This is also fine, but it is often useful to review the changes and be able to write a commit message on multiple lines.
Commit messages should use an imperative language to describe their changes. Reading commits should be like reading a log of commands.
[How to write good commit messages.](https://chris.beams.io/posts/git-commit/)
Commits should also be easy to understand and it is best to package changes into many well described commits than into one commit doing multiple things.

#### Pushing local commits

Now you put your local changes into commits and they are part of your local commit history. You can inspect it using `git log`.
To push your commits to the remote repository use, you guessed it:
```
$ git push
```
Your git identity will need to have write access rights to the remote repository to write to it, so make sure you set up your git identity correctly.
When your commits are pushed, the remote repository is updated and other contributers can pull the new version.

#### Branching

The commit history is a chain of commits managed by git. It is possible to "branch" out a second commit history
to be able to work on different versions of the same project in parallel. This is can be visualized like this:
```
        * - * <- feature branch
      / C   E
* - * - * - * <- master branch
A   B   D   F
```
`A`, `B`, `C`, etc are commits. `master` and `feature` are branches.
As you can see, both branches share some commits (`A` and `B`) but *diverge* at some point and contain different commits.
When working in git, you are always "on" a branch. You can switch branches using `git checkout branch_name`.
Git will then change the contents of your local repository's directory to match the branches' version of the repository.

Branches can later be merged back together by using `git merge`, but this usually creates an ugly "merge commit",
where any conflicts between the two branches are fixed. It is better to rebase one of the branches first, and then merge by "fast-forwarding":
- First we change to the branch we want to merge into `master`, in this example it is called `feature`
```
$ git checkout feature
```
- Then we rebase `feature` onto `master`
```
$ git rebase master
```
- We fix any conflicts as our `feature` commits are applied to the end of `master`. The fixes will be part of the commits that made the breaking changes.
- Now our `feature` branch looks just like `master` with some extra commits at the end:
```
               * - * <- feature branch
             / C   E
* - * - * - * <- master branch
A   B   D   F
```
- We change back to `master`: `git checkout master`
- and we merge `feature` without a merge commit:
```
git merge feature --ff-only
```
(`--ff-only` means fast-forward only. If for some reason fast-forward is not possible we will get an error message.
This can be set to default using `git config --global merge.ff only`)
- Now `master` contains the extra commits from `feature`:
```
               * - * <- feature branch
             / C   E
* - * - * - * - * - * <- master branch
A   B   D   F   C   E
```
- The feature branch can now be deleted
```
$ git branch -D feature
```
- to delete a remote branch:
```
$ git push origin -d feature
```

This should basically be all you need to know to use git!

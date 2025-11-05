# Contributing to Crispo

First off, thank you for considering contributing to Crispo! It's people like you that make Crispo such a great tool.

## Where do I go from here?

If you've noticed a bug or have a feature request, [make one](https://github.com/LOOFYYLO/crispo/issues/new)! It's generally best if you get confirmation of your bug or approval for your feature request this way before starting to code.

### Fork & create a branch

If this is something you think you can fix, then [fork Crispo](https://github.com/LOOFYYLO/crispo/fork) and create a branch with a descriptive name.

A good branch name would be (where issue #33 is the ticket you're working on):

```bash
git checkout -b 33-add-new-optimizer
```

### Get the test suite running

Make sure you're running the tests before you make any changes. You can run the tests with:

```bash
python -m unittest test_crispo.py
```

### Implement your fix or feature

At this point, you're ready to make your changes! Feel free to ask for help; everyone is a beginner at first :smile_cat:

### Make a Pull Request

At this point, you should switch back to your master branch and make sure it's up to date with Crispo's master branch:

```bash
git remote add upstream git@github.com:LOOFYYLO/crispo.git
git checkout master
git pull upstream master
```

Then update your feature branch from your local copy of master, and push it!

```bash
git checkout 33-add-new-optimizer
git rebase master
git push --force-with-lease origin 33-add-new-optimizer
```

Finally, go to GitHub and [make a Pull Request](https://github.com/LOOFYYLO/crispo/compare)

### Keeping your Pull Request updated

If a maintainer asks you to "rebase" your PR, they're saying that a lot of code has changed, and that you need to update your branch so it's easier to merge.

To learn more about rebasing and merging, check out [this guide](https://www.atlassian.com/git/tutorials/merging-vs-rebasing).

Once you've updated your branch, you'll need to force push the changes to your remote branch.

```bash
git push --force-with-lease origin 33-add-new-optimizer
```

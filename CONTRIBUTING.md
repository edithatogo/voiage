# Contributing to voiage

First off, thank you for considering contributing to `voiage`. It's people like you that make `voiage` such a great tool.

## Where do I go from here?

If you've noticed a bug or have a question, [search the issue tracker](https://github.com/doughnut/voiage/issues) to see if someone else has already created a ticket. If not, go ahead and [create a new one](https://github.com/doughnut/voiage/issues/new)!

## Fork & create a branch

If you're looking to contribute code, the first step is to fork this repo. Then, create a branch with a descriptive name.

A good branch name would be (where issue #33 is the ticket you're working on):

```bash
git checkout -b 33-add-a-new-feature
```

## Get the project running

Once you've forked and cloned the repo, you'll need to get the project running. This project uses `pip` to manage dependencies. To install the dependencies, run:

```bash
pip install -e .[dev]
```

## Make your changes

Now you're ready to make your changes!

## Test your changes

Once you've made your changes, you'll need to test them. To run the tests, run:

```bash
pytest
```

## Lint your changes

This project uses `ruff` to lint the code. To lint your changes, run:

```bash
ruff check .
```

## Commit your changes

Once you're happy with your changes, you'll need to commit them. Make sure to write a good commit message.

## Push your changes

Once you've committed your changes, you'll need to push them to your fork.

## Create a pull request

Once you've pushed your changes, you'll need to create a pull request. Make sure to link the pull request to the issue you're working on.

## Wait for a review

Once you've created a pull request, you'll need to wait for a review. Once your pull request has been approved, it will be merged into the `main` branch.

## Celebrate!

You've just contributed to `voiage`! Thank you for your hard work.

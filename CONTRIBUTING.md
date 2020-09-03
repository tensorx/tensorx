# Contributing to TensorX
First off, thanks for taking the time to contribute!

These are mostly contribution guidelines, not rules. 
Feel free to propose changes to this document in a pull request.

### Table of contents

[How to contribute](#how-to-contribute)
  
  * [Reporting bugs](#reporting-bugs)
  * [Suggesting enhancements](#suggesting-enhancements)
  * [Contributing to code](#contributing-to-code)
  
## How to contribute

### Reporting bugs

Bugs are tracked on the official [issue tracker](https://github.com/davidenunes/tensorx/issues).
When you are creating a bug report, please include as many details as possible. 
    
   * Use a clear and descriptive title for the issue to identify the problem.
   * Describe the exact steps which reproduce the problem in as many details as possible.
   * Provide specific examples to demonstrate the steps to reproduce the issue. Include links to files or GitHub projects, or snippets, which you use in those examples.
   * Describe the behavior you observed after following the steps and point out what exactly is the problem with that behavior.
   * **Explain which behavior you expected to see instead and why.**
   * Make sure you have searched the [issues](https://github.com/davidenunes/tensorx/issues) to make sure the bug is not a duplicate.
   * If you find a Closed issue that matches the problem you are experiencing, open a new issue and include a link to the original issue.

Provide more **context** by answering these questions:
    
   * Did the problem start happening recently (e.g. after updating to a new version of TensorX) or was this always a problem?
   * If the problem started happening recently, can you reproduce the problem in an older version of TensorX? What's the most recent version in which the problem does not happen?
   * Can you reliably reproduce the issue? If not, provide details about how often the problem happens and under which conditions it normally happens.

Include details about your configuration and environment:

   * Which version of TensorX are you using? 
   * Which version of Tensorflow are you using
   * Which Python version TensorX has been installed for? 
   * What's the name and version of the OS you're using?

### Suggesting enhancements

Suggestions are also tracked on the official [issue tracker](https://github.com/davidenunes/tensorx/issues).

* Make sure you have searched the [issues](https://github.com/davidenunes/tensorx/issues) to make sure your suggestion is not a duplicate.
* Use a clear and descriptive title for the issue to identify the suggestion.
* Provide a step-by-step description of the suggested enhancement in as many details as possible.
* Provide specific use cases for your suggestion.
* Describe the current behavior and explain which behavior you expected to see instead and why.


### Contributing to code

I recommend using [Poetry](https://python-poetry.org/docs/#introduction) to setup a local 
environment where you can contribute to and test TensorX codebase. You will first need to clone the repository using git and place yourself in its directory:

```bash
git clone git@github.com:davidenunes/tensorx.git
cd tensorx
```
With _Poetry_ installed, you will need to create a new _virtualenv_ where the required dependencies for TensorX will be installed. 
After that, be sure that the current tests are passing on your machine:

```bash
poetry install
poetry run pytest tests/
```

#### Pull Requests

* Include unit tests when you change code or contribute new features, as they help to a) prove that your code works correctly, and b) guard against future breaking changes to lower the maintenance cost.
* Bug fixes require unit tests. The presence of a bug usually indicates insufficient test coverage.
* Add/Update documentation for the contributed code. _docstrings_ use the [Google style](https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings).

Additional Considerations: 

* When you contribute a new feature to TensorX, the maintenance burden is transferred to me. Unless you wish to become a regular maintainer, keep this maintenance cost in mind.
* Avoid Pull Requests to fix one typo, one warning,etc. I recommend fixing the same issue at the file level at least.


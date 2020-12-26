# Contributing to TensorX
First off, thanks for taking the time to contribute!

These are mostly contribution guidelines, not rules. 
Feel free to propose changes to this document in a pull request.

## Community
If you want to stay up to date with what is going on in the project, want to discuss features and use cases, 
talk about fixes or simply ask a question you can join the following channels:

* Users mailing list: [https://groups.google.com/g/tensorx-users](https://groups.google.com/g/tensorx-users)
* Discord Server: [https://discord.gg/ZNeeQQHa](https://discord.gg/ZNeeQQHa)
  
## Reporting bugs

Bugs are tracked on the official [issue tracker](https://github.com/davidenunes/tensorx/issues).
When you are creating a bug report, please include as many details as possible. 
    
   * Use a clear and descriptive title for the issue to identify the problem;
   * Make sure you have searched the [issues](https://github.com/davidenunes/tensorx/issues) to make sure the bug is not a duplicate;
   * If you find a Closed issue that matches the problem you are experiencing, open a new issue and include a link to the original issue;
   * Describe the exact steps which reproduce the problem in as many details as possible;
   * Include details about your configuration and environment;
   * Provide specific examples to demonstrate the steps to reproduce the issue;
   * Describe the behavior you observed after following the steps and point out what exactly is the problem with that behavior;
   * **Explain which behavior you expected to see instead and why.**
   

## Suggesting enhancements

Suggestions are also tracked on the official [issue tracker](https://github.com/davidenunes/tensorx/issues).

* Make sure you have searched the [issues](https://github.com/davidenunes/tensorx/issues) to make sure your suggestion is not a duplicate.
* Use a clear and descriptive title for the issue to identify the suggestion.
* Provide a step-by-step description of the suggested enhancement in as many details as possible.
* Provide specific use cases for your suggestion.
* Describe the current behavior and explain which behavior you expected to see instead and why.


## Contributing to code

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

### Pull Requests

* Include unit tests when you change code or contribute new features, as they help to a) prove that your code works correctly, and b) guard against future breaking changes to lower the maintenance cost.
* Bug fixes require unit tests. The presence of a bug usually indicates insufficient test coverage.
* Add/Update documentation for the contributed code. _docstrings_ use the [Google style](https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings).


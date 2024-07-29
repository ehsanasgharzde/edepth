# Contributing to edepth

Thank you for considering contributing to edepth! Your help is highly appreciated. This document provides guidelines and instructions on how to contribute to the project.

## How to Contribute

### Reporting Bugs

If you encounter any issues or bugs, please report them by opening an issue on GitHub. Make sure to include the following information:

- A clear and descriptive title
- A detailed description of the issue
- Steps to reproduce the issue
- Any relevant logs or error messages
- Your environment setup (OS, Python version, etc.)

### Suggesting Features

We welcome suggestions for new features. To suggest a feature, please open an issue on GitHub and include the following details:

- A clear and descriptive title
- A detailed description of the proposed feature
- Any relevant use cases or examples

### Submitting Pull Requests

To submit a pull request, follow these steps:

1. **Fork the Repository**: Fork the edepth repository to your GitHub account.

2. **Create a New Branch**: Create a new branch for your feature or bug fix.
    ```bash
    git checkout -b feature-branch
    ```

3. **Commit Your Changes**: Make your changes and commit them with a clear and descriptive commit message.
    ```bash
    git commit -am 'Add new feature'
    ```

4. **Push to the Branch**: Push your changes to the new branch on your forked repository.
    ```bash
    git push origin feature-branch
    ```

5. **Create a Pull Request**: Open a pull request from your forked repository's branch to the main branch of the edepth repository. Provide a detailed description of your changes and any relevant context.

## Coding Conventions and Standards

Please adhere to the following coding standards to maintain code quality and consistency:

- Follow PEP 8 guidelines for Python code.
- Write clear and descriptive commit messages.
- Include comments and docstrings to explain your code.
- Ensure your code is well-organized and modular.

## Setting Up the Development Environment

To set up the development environment, follow these steps:

1. **Clone the Repository**: Clone the edepth repository to your local machine.
    ```bash
    git clone https://github.com/ehsanasgharzde/edepth.git
    cd edepth
    ```

2. **Create a Virtual Environment**: Create and activate a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install Dependencies**: Install the required dependencies from the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

## Running Tests

Before submitting a pull request, ensure that your changes do not break any existing functionality by running the tests. To run the tests, use the following command:

```bash
pytest
```

## Contact

For any questions, suggestions, or collaboration opportunities, feel free to reach out to [me](mailto:ehsanasgharzadeh.asg@gmail.com).

We look forward to your contributions!

Thank you,
Ehsan Asgharzadeh

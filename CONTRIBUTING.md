# Contributing to Pneumonia Detection AI

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Report unethical conduct to maintainers

## Getting Started

1. **Fork** the repository
2. **Clone** your fork: `git clone https://github.com/yourusername/pneumonia-detection-ai.git`
3. **Create** a branch: `git checkout -b feature/your-feature`
4. **Install** dev dependencies: `pip install -r requirements-dev.txt`
5. **Make** your changes
6. **Test** your code: `pytest tests/`
7. **Format** code: `black app/ tests/`
8. **Lint** code: `flake8 app/ tests/`
9. **Commit**: `git commit -m "Add your feature"`
10. **Push**: `git push origin feature/your-feature`
11. **Create** a Pull Request

## Code Quality Standards

### Python Style
- Follow PEP 8
- Use type hints
- Write docstrings for all functions
- Max line length: 100 characters

### Testing
- Write tests for new features
- Maintain >80% code coverage
- All tests must pass before merge

### Documentation
- Update README for new features
- Document API changes
- Add examples for complex functions

## Areas for Contribution

### High Priority
- 🐛 **Bug Fixes**: Report and fix issues
- 📊 **Performance**: Optimize inference speed
- 🧪 **Testing**: Increase test coverage

### Medium Priority
- 📝 **Documentation**: Improve guides and examples
- 🎨 **UI/UX**: Enhance user interface
- ♻️ **Refactoring**: Improve code structure

### Nice to Have
- 🌍 **Internationalization**: Multi-language support
- 📱 **Mobile**: Mobile app development
- 🔐 **Security**: Security hardening

## Reporting Issues

### Bug Reports
Include:
- Python version
- Environment (OS, Docker, etc.)
- Steps to reproduce
- Expected vs actual behavior
- Error messages/logs

### Feature Requests
Include:
- Use case
- Proposed solution
- Alternatives considered
- Implementation complexity

## Pull Request Process

1. Update documentation
2. Ensure tests pass: `pytest`
3. Update CHANGELOG.md
4. Reference related issues: `Closes #123`
5. Request review from maintainers

### PR Title Format
```
[CATEGORY] Brief description

Category: Feature, Fix, Docs, Test, Refactor, Performance
```

### PR Description Template
```markdown
## Description
Brief explanation of changes

## Related Issues
Closes #123

## Changes Made
- Change 1
- Change 2

## Testing
How to test the changes

## Screenshots (if applicable)
```

## Development Setup

### Install Development Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks (optional)
pre-commit install
```

### Running Tests
```bash
# All tests
pytest

# With coverage
pytest --cov=app tests/

# Specific test file
pytest tests/test_model.py

# Verbose output
pytest -v
```

### Code Quality Tools
```bash
# Format code
black app/

# Check style
flake8 app/

# Type checking
mypy app/

# All in one
black app/ && flake8 app/ && mypy app/
```

## Commit Guidelines

### Format
```
[TYPE] Subject (50 chars or less)

Optional body (wrap at 72 chars)

Fixes #123
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test additions/changes
- `chore`: Build, dependencies

### Example
```
feat: Add batch prediction script

- Implement batch inference
- Add progress bar
- Include error handling

Fixes #45
```

## Documentation

### Adding a New Feature
1. Update docstrings with examples
2. Add API documentation
3. Update README if user-facing
4. Create usage examples

### Writing Good Docstrings
```python
def example_function(param1: str, param2: int) -> bool:
    """
    Brief one-line description.
    
    Longer description explaining the function's purpose
    and behavior in detail.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When something is invalid
        
    Example:
        >>> example_function("test", 42)
        True
    """
    pass
```

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Contact

- **Issues**: Use GitHub Issues for bugs and features
- **Discussions**: Use GitHub Discussions for general questions
- **Email**: contact@example.com

---

Thank you for contributing to making AI healthcare more accessible! 🫁

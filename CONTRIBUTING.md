# Contributing to Bespoke Decision Tree Utility

Thank you for your interest in contributing to the Bespoke Decision Tree Utility! We welcome contributions from the community and are grateful for your support.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Submitting Changes](#submitting-changes)
- [Issue Guidelines](#issue-guidelines)
- [Pull Request Process](#pull-request-process)
- [Community Guidelines](#community-guidelines)

## 🤝 Code of Conduct

This project and everyone participating in it is governed by our commitment to creating a welcoming and inclusive environment. By participating, you are expected to uphold these values:

### Our Standards

- **Be respectful**: Treat all community members with respect and kindness
- **Be inclusive**: Welcome newcomers and help them feel part of the community  
- **Be collaborative**: Work together constructively and share knowledge
- **Be patient**: Understand that people have different skill levels and backgrounds
- **Be constructive**: Provide helpful feedback and suggestions

### Unacceptable Behavior

- Harassment, discrimination, or offensive comments
- Personal attacks or inflammatory language
- Publishing private information without consent
- Other conduct deemed inappropriate in a professional setting

## 🚀 Getting Started

### Prerequisites

Before contributing, ensure you have:

- Python 3.8 or higher
- Git installed and configured
- Familiarity with PyQt5 and decision tree algorithms (helpful but not required)
- Basic understanding of the project structure

### First Steps

1. **Fork the Repository**
   ```bash
   # Click the 'Fork' button on GitHub, then clone your fork
   git clone https://github.com/rupinrd3/bespoke-decision-tree-utility.git
   cd bespoke-decision-tree-utility
   ```

2. **Set Up Development Environment**
   ```bash
   # Create a virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Install development dependencies
   pip install -r requirements-dev.txt  # If available
   ```

3. **Verify Setup**
   ```bash
   # Run the application
   python main.py
   
   # Run tests (if available)
   python -m pytest tests/
   ```

## 🛠️ How to Contribute

There are many ways to contribute to this project:

### 🐛 Bug Reports

Help us improve by reporting bugs:
- Check existing issues first to avoid duplicates
- Use the bug report template
- Provide detailed steps to reproduce
- Include system information and error messages

### ✨ Feature Requests

Suggest new features or improvements:
- Check if the feature already exists or is planned
- Use the feature request template
- Explain the use case and benefits
- Consider implementation complexity

### 📝 Documentation

Improve project documentation:
- Fix typos and unclear explanations
- Add examples and tutorials
- Update setup instructions
- Translate documentation

### 💻 Code Contributions

Contribute code improvements:
- Bug fixes
- New features
- Performance optimizations
- UI/UX enhancements
- Test coverage improvements

## 🔧 Development Setup

### Project Structure

```
bespoke-decision-tree-utility/
├── main.py                     # Application entry point
├── analytics/                 # Statistical analysis modules
├── business/                  # Business logic 
├── data/                      # Data processing
├── export/                    # Model export functionality
├── models/                    # Core algorithms
├── ui/                        # User interface
│   ├── components/           # Reusable UI components
│   ├── dialogs/              # Dialog windows
│   └── widgets/              # Custom widgets
├── utils/                     # Utility functions
├── workflow/                  # Workflow engine
└── tests/                     # Test files
```

### Key Technologies

- **UI Framework**: PyQt5
- **Data Processing**: pandas, NumPy
- **Machine Learning**: scikit-learn
- **Visualization**: matplotlib, seaborn
- **Export**: PMML, JSON serialization

## 📏 Coding Standards

### Python Style Guide

We follow PEP 8 with some modifications:

```python
# Use descriptive variable names
def calculate_node_statistics(node_data, target_variable):
    """Calculate comprehensive statistics for a tree node.
    
    Args:
        node_data: DataFrame containing node records
        target_variable: Name of the target column
        
    Returns:
        dict: Dictionary containing calculated statistics
    """
    pass

# Use type hints where helpful
from typing import List, Dict, Optional

def find_optimal_split(data: pd.DataFrame, 
                      target: str) -> Optional[Dict[str, float]]:
    """Find the optimal split for the given data."""
    pass
```

### Code Organization

- **Modularity**: Keep functions and classes focused on single responsibilities
- **Documentation**: Use docstrings for all public functions and classes
- **Error Handling**: Include appropriate try/catch blocks and user-friendly error messages
- **Logging**: Use the existing logging framework for debug information

### UI Guidelines

- **Consistency**: Follow existing UI patterns and styling
- **Accessibility**: Ensure UI elements are keyboard accessible
- **Responsiveness**: Test UI with different window sizes
- **Internationalization**: Use translatable strings where possible

## 🧪 Testing Guidelines

### Test Structure

```python
import pytest
from unittest.mock import Mock, patch
from models.decision_tree import DecisionTree

class TestDecisionTree:
    def setup_method(self):
        """Set up test fixtures."""
        self.sample_data = create_sample_dataset()
        
    def test_tree_building(self):
        """Test basic tree building functionality."""
        tree = DecisionTree()
        result = tree.build_tree(self.sample_data, target='outcome')
        assert result is not None
        assert tree.get_depth() > 0
```

### Testing Requirements

- **Unit Tests**: Test individual functions and methods
- **Integration Tests**: Test component interactions
- **UI Tests**: Test user interface functionality (where applicable)
- **Performance Tests**: Ensure large dataset handling
- **Cross-Platform Tests**: Verify Windows and Ubuntu compatibility

## 📤 Submitting Changes

### Branch Naming Convention

- `feature/description` - New features
- `bugfix/issue-number` - Bug fixes
- `enhancement/description` - Improvements
- `docs/description` - Documentation updates

### Commit Message Format

```
type(scope): brief description

Longer explanation if needed. Wrap at 72 characters.

- List specific changes
- Reference issue numbers: Fixes #123
- Include breaking change notes if applicable
```

**Types**: feat, fix, docs, style, refactor, test, chore

**Examples**:
```
feat(tree): add support for categorical split grouping

- Implement multi-category binning algorithm
- Add UI controls for category selection
- Update export formats to include groupings
- Fixes #45

fix(ui): resolve tree visualization rendering issue

The tree nodes were not displaying correctly on high-DPI screens.
Updated the scaling calculations in tree_visualizer.py.

Fixes #67
```

## 🐛 Issue Guidelines

### Bug Reports

Use this template for bug reports:

```markdown
**Bug Description**
A clear description of the issue.

**Steps to Reproduce**
1. Go to '...'
2. Click on '...'
3. See error

**Expected Behavior**
What should happen.

**Actual Behavior**
What actually happens.

**Environment**
- OS: [e.g., Windows 10, Ubuntu 20.04]
- Python version: [e.g., 3.8.5]
- Application version: [e.g., 4.47]

**Additional Context**
Screenshots, error logs, or other relevant information.
```

### Feature Requests

Use this template for feature requests:

```markdown
**Feature Description**
A clear description of the proposed feature.

**Use Case**
Why is this feature needed? What problem does it solve?

**Proposed Solution**
How might this feature work?

**Alternatives Considered**
Other approaches you've considered.

**Additional Context**
Mockups, examples, or related issues.
```

## 🔄 Pull Request Process

### Before Submitting

1. **Ensure Tests Pass**: Run all tests and ensure they pass
2. **Update Documentation**: Update relevant documentation
3. **Follow Style Guide**: Ensure code follows project conventions
4. **Check Dependencies**: Minimize new dependencies
5. **Test Cross-Platform**: Verify changes work on Windows and Ubuntu

### Pull Request Template

```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that causes existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] Added tests for new functionality
- [ ] Tested on Windows
- [ ] Tested on Ubuntu

## Related Issues
Fixes #(issue number)

## Screenshots (if applicable)
Add screenshots for UI changes.
```

### Review Process

1. **Automated Checks**: CI/CD pipeline runs automatically
2. **Code Review**: Maintainers review the code
3. **Feedback**: Address any feedback or requested changes
4. **Approval**: Changes are approved by maintainers
5. **Merge**: Pull request is merged into the main branch

## 🌟 Community Guidelines

### Getting Help

- **Documentation**: Check existing documentation first
- **Issues**: Search existing issues for similar problems
- **Discussions**: Use GitHub Discussions for questions
- **Discord**: Join our Discord server for real-time chat (if available)

### Recognition

Contributors are recognized in several ways:
- Listed in the CONTRIBUTORS.md file
- Mentioned in release notes for significant contributions
- Eligible for contributor badges and recognition

### Mentorship

New contributors can:
- Look for "good first issue" labels
- Ask for mentorship in discussions
- Pair program with experienced contributors
- Join community calls and workshops

## 📞 Contact

- **Project Maintainers**: [List maintainer GitHub usernames]
- **Email**: support@bespoke-analytics.com
- **Discord**: [Discord server link if available]
- **Discussions**: [GitHub Discussions link]

---

## 🙏 Thank You

Thank you for contributing to the Bespoke Decision Tree Utility! Your contributions help make this tool better for the entire data science community.

**Happy coding!** 🎉
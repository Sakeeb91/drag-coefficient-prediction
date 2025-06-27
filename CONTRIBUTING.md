# Contributing to Physics-Guided Neural Network for Drag Coefficient Prediction

🎉 Thank you for considering contributing to this project! This document provides guidelines for contributing to make the process smooth and productive for everyone.

## 🚀 Quick Start

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Create a branch** for your feature or fix
4. **Make your changes** with clear commit messages
5. **Test your changes** thoroughly
6. **Submit a pull request** with a detailed description

## 🎯 Types of Contributions

We welcome various types of contributions:

### 🔬 **Scientific Enhancements**
- Extending to new physics domains (non-spherical objects, compressible flow)
- Adding new empirical correlations
- Implementing Physics-Informed Neural Networks (PiNNs)
- Validation with experimental/CFD data

### 💻 **Technical Improvements**
- Code optimization and performance improvements
- New model architectures (CNNs, RNNs, Transformers)
- Better visualization tools
- API development (FastAPI, Flask)
- Web interface (Streamlit, Dash)

### 📚 **Documentation**
- Improving README clarity
- Adding code comments
- Creating tutorials or examples
- Writing scientific explanations

### 🐛 **Bug Fixes**
- Fixing identified issues
- Improving error handling
- Adding input validation

## 📋 Development Guidelines

### **Environment Setup**
```bash
# Clone your fork
git clone https://github.com/YOUR-USERNAME/drag-coefficient-prediction.git
cd drag-coefficient-prediction

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (if available)
pip install -e .
```

### **Code Standards**
- Follow **PEP 8** style guidelines
- Use meaningful variable names
- Add docstrings to functions and classes
- Keep functions focused and modular
- Include type hints where appropriate

### **Testing**
- Test your changes thoroughly
- Ensure the model still achieves good performance
- Verify visualizations render correctly
- Check that physics validation still passes

### **Commit Messages**
Use clear, descriptive commit messages:
```
Add support for ellipsoidal particle drag prediction

- Implement new correlation for ellipsoid drag coefficients
- Add aspect ratio as input parameter
- Update visualization to show shape effects
- Maintain backwards compatibility with spherical particles
```

## 🔬 Scientific Contribution Guidelines

### **Physics Validation**
Any new physics implementations must:
- Include proper unit checking
- Validate against known analytical solutions
- Provide appropriate references to literature
- Maintain dimensional consistency

### **Data Quality**
When adding new datasets:
- Document data sources clearly
- Include uncertainty estimates where available
- Validate against experimental data when possible
- Maintain consistent formatting

### **Model Performance**
New models should:
- Achieve comparable or better performance
- Include proper cross-validation
- Document computational requirements
- Provide comparison with baseline methods

## 📈 Pull Request Process

### **Before Submitting**
- [ ] Test your changes locally
- [ ] Update documentation if needed
- [ ] Add tests for new functionality
- [ ] Verify physics validation still passes
- [ ] Check that visualizations work correctly

### **Pull Request Template**
```markdown
## 🎯 Purpose
Brief description of what this PR accomplishes.

## 🔬 Changes Made
- List of specific changes
- New features added
- Bug fixes included

## 🧪 Testing
- How you tested the changes
- Performance impact (if any)
- Screenshots for visual changes

## 📚 Documentation
- Documentation updates included
- New examples or tutorials added

## ✅ Checklist
- [ ] Code follows project style guidelines
- [ ] Tests pass locally
- [ ] Documentation updated
- [ ] Physics validation maintained
```

## 🎓 Learning Resources

### **Physics Background**
- **Fluid Mechanics**: White, "Fluid Mechanics" 8th Edition
- **Drag Theory**: Morrison, "An Introduction to Fluid Mechanics"
- **Dimensional Analysis**: Buckingham Pi theorem

### **Machine Learning**
- **Neural Networks**: Goodfellow et al., "Deep Learning"
- **Physics-ML**: Karniadakis et al., "Physics-informed machine learning"
- **Scikit-learn**: Official documentation and tutorials

### **Python Best Practices**
- **Code Style**: PEP 8 Style Guide
- **Documentation**: NumPy documentation style
- **Testing**: pytest framework

## 🤝 Code of Conduct

### **Our Standards**
- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Acknowledge contributions from others

### **Unacceptable Behavior**
- Harassment or discrimination
- Trolling or insulting comments
- Personal attacks
- Publishing private information

## 🆘 Getting Help

### **Questions?**
- **GitHub Issues**: For technical questions and bug reports
- **Discussions**: For general questions and ideas
- **Email**: For private matters (create GitHub issue for contact)

### **Stuck?**
Don't hesitate to ask for help! We're here to support contributors at all levels.

## 🎉 Recognition

Contributors will be:
- Listed in the project README
- Mentioned in release notes
- Credited in any resulting publications (for significant scientific contributions)

## 📝 License

By contributing, you agree that your contributions will be licensed under the same MIT License that covers the project.

---

Thank you for contributing to advancing physics-guided machine learning! 🚀🔬
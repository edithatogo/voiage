# Security Policy for voiage

## Supported Versions

We release patches for security vulnerabilities. The following versions are supported:

| Version | Supported          |
| ------- | ------------------ |
| 0.2.x   | ✅ Yes             |
| < 0.2   | ❌ No              |

## Reporting a Vulnerability

If you discover a security vulnerability in voiage, please report it responsibly by sending an email to [security@voiage.org](mailto:security@voiage.org) or by creating a private security advisory on GitHub.

Please do not report security vulnerabilities through public GitHub issues.

When reporting a vulnerability, please include:
- A clear description of the vulnerability
- Steps to reproduce the issue
- The potential impact of the vulnerability
- Any suggested fixes if known

## Security Measures

The voiage project implements the following security measures:

- **Dependency Scanning**: Regular scanning for known vulnerabilities in dependencies using Safety
- **Code Scanning**: Security analysis with Bandit to detect potential security issues in the codebase
- **Automated Security Checks**: Integration of security checks in CI/CD pipeline
- **Regular Updates**: Dependabot automatically opens pull requests for security updates
- **Pre-commit Hooks**: Security checks run locally before commits

## Security Testing

Security checks are performed through:

1. **Static Analysis**: Bandit is used for static analysis of Python code
2. **Dependency Check**: Safety checks for known vulnerabilities in dependencies
3. **CI Integration**: Security checks run in the CI pipeline on each pull request and merge to main

## Security Updates

- Security updates and patches are released as soon as possible after vulnerabilities are discovered
- Users are encouraged to keep their voiage installations up-to-date
- Critical security updates will be announced via GitHub releases and PyPI

## Best Practices for Users

When using voiage in your projects, consider:

1. Keep your voiage installation updated
2. Use virtual environments to isolate dependencies
3. Regularly audit your dependencies
4. Follow secure coding practices in your applications
5. Be mindful of data privacy and security requirements applicable to your use case

## Acknowledgments

We thank the security researchers and community members who responsibly report security vulnerabilities and help improve the security of this project.

## Additional Resources

- [PyPA Security Best Practices](https://packaging.python.org/guides/publishing-package-distribution-releases-and-certifying-signatures/#security-considerations)
- [OWASP Python Security Guidelines](https://owasp.org/www-project-top-ten/)
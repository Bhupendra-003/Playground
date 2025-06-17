# OWASP Methodology

The OWASP Testing Guide (WSTG) is globally recognized framework for conducting security testing for web applications.
- It Breaks down process into structured phases.
- Each phase contains test cases and best practices to ensure thorough coverage.

## Tool Used
- OWASP Testing Guide (WSTG v4)
- Burp Suite Professional
- OWASP ZAP
- Recon
- Browser Developer Tools

## Procedure
- Define Scope - Idetify web application, its functionality and define boundaries of testing.
- Perform Reconnaissance - Use passive and active tool to gather target info.
- Configure Testing - Examine http headers, SSL certificates and software versions.
- Authentication Testing - Check Login Forms, Session Timeout, Forgot Password flow.
- Authorization Testing - Check acces control for various roles by manipulating requiests.
- Input Validation - Inject Crafted payloads into form fields to test for XSS, SQLi and command injection.
- Session Management - Test for secure cookie and handling, session fixation and token reuse.
- Buisness Logic Testing - Attempt to abuse workflows or bypass steps in buisness functions.
- Documentation - Record findings with evidence ( screenshots, logs) classify by severity and provide remediation.
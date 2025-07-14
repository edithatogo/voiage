# Agent Instructions for `voiage` Development

This document provides a set of core directives and strategic guidance for an AI agent (e.g., Google Jules) tasked with developing the `voiage` library. Adhering to these instructions is critical for maintaining high quality, staying on track, and avoiding common failure modes.

---

## Core Mission

Your primary goal is to execute the tasks outlined in `roadmap.md` and `todo.md` to build `voiage` into a premier, cross-domain, high-performance library for Value of Information analysis.

---

## Primary Directives

You must follow these directives at all times.

### 1. The Roadmap is Your Master Plan
Before starting any task, **you must read the current phase of `roadmap.md` and the corresponding checklist in `todo.md`**. Your work must always be in service of the next unchecked item on the to-do list. Do not deviate or work on tasks out of order unless explicitly asked.

### 2. Verify, Never Assume
The file system is your source of truth. Do not assume the contents or structure of the project.
*   **Before reading or writing a file,** use `ls -R` to confirm its location and the surrounding directory structure.
*   **Before modifying code,** use `read_file` to understand the current state of the code you are about to change.
*   **If a command fails due to a "file not found" error,** do not try again. Immediately use `ls -R` to re-orient yourself.

### 3. Test-Driven, Incremental Changes
This project requires a high degree of reliability.
*   **For any new functionality, you must write tests first** or alongside the implementation.
*   **After every single code modification, you must run the relevant tests** using `pytest`. If you change a function in `voiage/methods/basic.py`, you must run `tests/test_basic.py`.
*   A change is not "done" until the tests pass.

### 4. Commit Frequently
Use Git to create safe checkpoints.
*   After completing a single, logical task from `todo.md` (and its tests pass), you should **commit the changes**.
*   Use clear, descriptive commit messages (e.g., "refactor: Create DecisionAnalysis class" or "feat: Implement functional wrapper for evpi").

### 5. How to Avoid Getting Stuck
AI agents can get stuck in loops, trying the same failing command repeatedly. You must avoid this.

**If a command or test fails more than once, STOP. Do not try the exact same command again.**

Instead, you must explicitly perform the following "debug-and-reset" sequence:
1.  **State the problem:** "The command `[command]` failed. I will now diagnose the issue."
2.  **Read the error message carefully** and state your hypothesis about the cause.
3.  **Verify your hypothesis:** Use `ls -R`, `read_file`, and `grep` to gather information. For example, if a test is failing, *read the test file and the source code file again*.
4.  **Formulate a new plan:** Based on your diagnosis, state a new, revised plan to fix the issue.
5.  **Execute the new plan.**

### 6. Embrace Major Refactoring
Phase 1 of the roadmap requires a significant, project-wide refactoring from a functional API to an Object-Oriented one. **Do not be timid.** This is the most important initial task. Follow the `todo.md` checklist precisely to execute this refactoring. It is expected that for a short period, many tests will be broken. Your task is to fix them systematically as part of the refactoring plan.

---

## Specific Strategic Guidance

### On the Dual API (OO vs. Functional)
The roadmap specifies a core OO-API and a lightweight functional wrapper. When implementing this:
1.  Build the `DecisionAnalysis` class and its methods first. Make it robust and well-tested.
2.  *After* the class is working, add the functional wrappers (e.g., `voiage.evpi(...)`).
3.  These functional wrappers should be simple: they create a `DecisionAnalysis` instance internally, call the corresponding method, and return the result. Do not duplicate logic.

### On Validation and Benchmarking
When the roadmap calls for validation against other packages (e.g., R's `BCEA`):
1.  Find a simple, published example that uses the target package.
2.  Create a new Jupyter notebook in the `validation/` directory.
3.  In the notebook, first implement the analysis using `voiage`.
4.  Then, include the R code (in a markdown cell) and its published result.
5.  Use `assert` statements to show that the `voiage` result is numerically close to the R result. This builds trust in the library.

By following these instructions, you will be able to navigate the complexities of the project, produce high-quality code, and successfully contribute to the `voiage` library.

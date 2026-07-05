---
name: docker-verify
description: Verify Dockerfile and compose configuration against organization standards.
---
# Docker Verify Protocol

1. **Audit Targets**: `Dockerfile` and `docker-compose.yml` within the service directory.
2. **Verification Checklist**:
   - **Base Image**: Must be the organization's approved hardened base image.
   - **User**: Must not run as `root` (User ID >= 1000).
   - **Healthchecks**: Must define `HEALTHCHECK` instructions.
   - **Labels**: Must include `maintainer` and `service_name` labels.
3. **Execution**:
   - Run `hadolint` (if available) on the `Dockerfile`.
   - Report any security or performance regressions.

4. Execution
- Trigger: `/docker-verify`
- Output: A report highlighting "Clean" components and "Violation" instances with suggestions for refactoring.

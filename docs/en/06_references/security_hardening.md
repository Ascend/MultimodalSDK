# Security Hardening

## Security Requirements

When you use an API to read a file, ensure that the file owner is the current user and that the file permissions are no more permissive than `640`. This helps prevent privilege escalation and other potential security issues.

Software code or programs downloaded from external sources may pose security risks. Users are responsible for ensuring the security of such software.

## Hardening Precautions

The security hardening measures listed in this document are basic recommendations. Users should re-evaluate the security hardening measures for the entire system based on their specific business requirements. When necessary, refer to industry best practices and consult security experts.

## OS Security Hardening

### PATH Configuration for Regular Users

After the OS is installed, if regular users are configured, you can add the `ALWAYS_SET_PATH=yes` configuration to the `/etc/login.defs` file to prevent unauthorized operations that could lead to privilege escalation.

### Setting umask

Users are advised to set the umask to `027` or more restrictive on the host and in containers to improve file security.

The following uses `027` as an example.

1. Log in to the server as the root user and edit the `/etc/profile` file.

   ```bash
   vim /etc/profile
   ```

2. Add `umask 027` to the end of the `/etc/profile` file, and then save and exit.

3. Run the following command to apply the configuration.

   ```bash
   source /etc/profile
   ```

### Ownerless File Hardening

Because the OS in official Docker images differs from that on the physical machine, system users may not have one-to-one mappings. As a result, files generated during physical machine or container operation may become ownerless.

You can run `find / -nouser -o -nogroup` to find ownerless files on the host or in containers. Create corresponding users and user groups based on the UIDs and GIDs of the files, or modify the UIDs of existing users and the GIDs of existing user groups to match. This assigns owners to the files and prevents security risks caused by ownerless files.

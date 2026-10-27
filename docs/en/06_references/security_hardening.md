# Security Hardening

## Security Requirements

When you use an API to read a file, ensure that you own the file and that its permissions are no more permissive than 640. This helps prevent privilege escalation and other security issues.

Software code or programs downloaded from external sources may pose risks. You must ensure that their functions are secure.

## Hardening Precautions

The security hardening measures listed in this document are basic hardening recommendations. You should re-evaluate the network security hardening measures for the entire system based on your own service, and when necessary, you can refer to industry best practices and the advice of security experts.

## OS Security Hardening

### Common User PATH Configuration

After installing the operating system, if you configure common users, you can add `ALWAYS_SET_PATH=yes` to the `/etc/login.defs` file to prevent unauthorized operations.

### Setting umask

You are advised to set the umask on the host and in containers to 027 or a more restrictive value to tighten file permissions.

The following example shows how to set the umask to 027.

1. Log in to the server as the root user and edit the `/etc/profile` file.

    ```bash
    vim /etc/profile
    ```

2. Add `umask 027` to the end of the `/etc/profile` file, then save and exit.
3. Run the following command to make the configuration take effect.

    ```bash
    source /etc/profile
    ```

### Ownerless File Hardening

Because official Docker images differ from the operating system on the host machine, the users in the system may not correspond one to one. Therefore, files created while the host machine or a container is running can become ownerless.

You can run the `find / -nouser -o -nogroup` command to locate ownerless files in a container or on the host machine. Create corresponding users and user groups based on the file UID and GID, or modify the UID of existing users and the GID of user groups to match the file. Then assign ownership of the files to prevent ownerless files from posing security risks to the system.

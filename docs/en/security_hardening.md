# Security Hardening

## Security Requirements

When you use an API to read a file, ensure that you own the file and that its permissions are no more permissive than 640. This helps prevent privilege escalation and other security issues.

Software code or programs downloaded from external sources may pose risks. You must ensure that their functions are secure.

## Hardening Precautions

The security hardening measures listed in this document are basic recommendations. You should re-evaluate the network security hardening of the entire system based on your own service requirements. When necessary, consult industry best practices and security experts.

## OS Security Hardening

### PATH Configuration for Regular Users

After installing the OS, if common users are configured, you can add `ALWAYS_SET_PATH yes` to the `/etc/login.defs` file to prevent unauthorized privilege escalation.

### Setting umask

Set the host umask to 027 or more restrictive on the host and in containers to enhance file security.

The following example shows how to set the umask to 027.

1. Log in to the server as the root user and edit the `/etc/profile` file.

    ```bash
    vim /etc/profile
    ```

2. Add `umask 027` to the end of the `/etc/profile` file, then save and exit.
3. Run the following command to apply the configuration.

    ```bash
    source /etc/profile
    ```

### Ownerless File Hardening

Differences between official Docker images and the host OS may result in a mismatch between user definitions. This can lead to the creation of ownerless files during system or container operation.

You can find ownerless files on the host or in containers by running `find / -nouser -o -nogroup`. To mitigate security risks, create corresponding users and groups based on file UIDs and GIDs, or adjust existing UIDs and GIDs to match. This ensures that every file has a valid owner and prevents ownerless files from creating security risks for the system.

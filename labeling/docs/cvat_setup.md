CVAT provides a docker compose file to set up everyting on a private machine, see instructions [here](https://cvat-ai.github.io/cvat/docs/administration/basics/installation/#ubuntu-1804-x86_64amd64). At the time of writing however, you need to change the container tag to `dev`, cf [this issue](https://github.com/opencv/cvat/issues/4816).


 You can also do this setup on a remote machine, in which case you can either make the client reachable over the web or forward the tcp conncetion to your local machine using ssh: `ssh -L <local-tcp-port>:<remote-machine>:<remote-cvat-port(8080 by default)> <user>@<remote-machine>`

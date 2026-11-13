# HiPerGator SSH over Tailscale

The Mac's `hpg.rc.ufl.edu` and `hpg-tail` SSH entries connect to
`hipergator.tailfed0c7.ts.net`. `HostKeyAlias hpg.rc.ufl.edu` and
`StrictHostKeyChecking yes` retain the trusted UF SSH host key. The first login
uses the normal UF password and Duo flow; later sessions reuse that connection.

Use a ControlPath independent of the Mac's local hostname for each route, such as
`~/.ssh/cm-%r@%h:%p`. The expanded paths must stay below the platform socket-path
limit (104 bytes on macOS).
When updating an existing installation, pin each route to the exact path reported
by `ssh -G` while `ssh -O check` confirms its live master. This preserves the live
socket while removing dependence on the Mac's changing local hostname. Keep the
socket directory private. `ControlPersist yes` removes the client idle limit;
`ConnectTimeout 15` bounds new TCP/handshake attempts without timing out Duo.

`hpg-public` connects directly to the public UF hostname. It may require password
and Duo when its old connection ends. Use it for recovery or bulk transfers when
the Tailscale path is relayed. Agent instructions explicitly selecting `hpg-public`
continue to use that route; such calls do not gain Tailscale roaming resilience.

## Recovery

If `ssh -fN hpg-tail` cannot reach the endpoint, use `ssh hpg-public`, then reach
`login7` internally and run `tsup`. Confirm the owner node with `hostname -s`.
Only then retry `ssh -fN hpg-tail` on the Mac. Do not start the same Tailscale
identity on another login or compute node: the home directory, socket and state
are shared. `hpg_tsup.bash` refuses a missing or different owner and waits for the
local daemon API before running the existing `tailscale up` command.

Install that function in the existing Tailscale section of HiPerGator's `.bashrc`;
retain its `TS_SOCK`, `tailscale` alias and `sshmac` alias. Record `login7` in
`~/.local/share/tailscale/owner` only after checking the live daemon on that node.
Changing the function does not restart the live daemon. If the owner node is
retired, reassignment requires first verifying that its daemon is stopped.

On the Mac, `TII_HPG_HOST="hpg-tail"` selects the shared connection. Set
`TII_RECOVERY_HINT` to include the public-route and owner-node recovery instructions.
The keeper never creates a login or answers an authentication prompt.

## Limits and checks

Brief network switches may recover while the same TCP connection and SSH master
remain alive. Keepalives disconnect after about five minutes of unanswered
checks. Reboot, forced sleep, remote daemon exit or key expiry can require a new
login. Idle-sleep prevention does not guarantee closed-lid operation.

The `hipergator` device key observed on 2026-11-13 expires on **2027-05-12**.
This date is an observation, not a setting to copy. Check current status before
recovery. No key-expiry or account setting is changed by this setup.

Userspace Tailscale can forward permitted incoming TCP connections to the login
node's loopback services. The effective filter observed on 2026-11-13 permits all
ports. Restricting access requires inspecting and deliberately changing the
actual tailnet policy; the startup helper does not change that policy.

Run the isolated helper tests with:

```sh
python3 -m unittest discover -s reports/research-advancement/keeper -p 'test_*.py' -v
```

For a live Wi-Fi test, record the master PID, hostname and existing socket first;
ensure a detached local recovery process turns Wi-Fi back on; interrupt Wi-Fi for
60 seconds; reconnect; verify a remote command, the same master PID and a keeper
`ok` check. A source or connectivity test alone does not establish roaming
survival. Sleep and reboot checks are separate and outside this guarantee.

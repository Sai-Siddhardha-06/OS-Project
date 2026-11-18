#!/usr/bin/env python3
import subprocess
import time
import sys
import os
import select

def run_qemu(commands, output_file=None, timeout=20, cpus=1, extra_args=None, stop_marker=None):
    xv6_dir = "/home/sai/OS Project/xv6-riscv"
    qemu_cmd = [
        'qemu-system-riscv64',
        '-machine', 'virt',
        '-bios', 'none',
        '-kernel', os.path.join(xv6_dir, 'kernel/kernel'),
        '-m', '128M',
        '-smp', str(cpus),
        '-nographic',
        '-global', 'virtio-mmio.force-legacy=false',
        '-drive', f'file={os.path.join(xv6_dir, "fs.img")},if=none,format=raw,id=x0',
        '-device', 'virtio-blk-device,drive=x0,bus=virtio-mmio-bus.0'
    ]
    if extra_args:
        qemu_cmd.extend(extra_args)

    p = subprocess.Popen(
        qemu_cmd,
        cwd=xv6_dir,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )

    # Set non-blocking on stdout
    os.set_blocking(p.stdout.fileno(), False)

    collected_bytes = bytearray()
    start_time = time.time()
    booted = False
    cmd_idx = 0
    last_cmd_time = 0

    while time.time() - start_time < timeout:
        if p.poll() is not None:
            break

        # Read available bytes without blocking
        try:
            chunk = os.read(p.stdout.fileno(), 4096)
            if chunk:
                collected_bytes.extend(chunk)
                current_text = collected_bytes.decode('utf-8', errors='replace')
                if ("$ " in current_text or "init: starting sh" in current_text) and not booted:
                    booted = True
                if stop_marker and stop_marker in current_text:
                    break
        except (BlockingIOError, OSError):
            pass

        # Send commands when booted
        if booted and cmd_idx < len(commands):
            if time.time() - last_cmd_time > 0.4:
                cmd = commands[cmd_idx] + '\n'
                p.stdin.write(cmd.encode('utf-8'))
                p.stdin.flush()
                cmd_idx += 1
                last_cmd_time = time.time()

        time.sleep(0.05)

    # Clean termination via Ctrl-A x (0x01, 'x')
    try:
        p.stdin.write(b'\x01x')
        p.stdin.flush()
    except Exception:
        pass

    try:
        rem, _ = p.communicate(timeout=2)
        if rem:
            collected_bytes.extend(rem)
    except subprocess.TimeoutExpired:
        p.kill()
        rem, _ = p.communicate()
        if rem:
            collected_bytes.extend(rem)

    out_str = collected_bytes.decode('utf-8', errors='replace')
    if output_file:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(out_str)
    return out_str

if __name__ == '__main__':
    cmds = sys.argv[1:] if len(sys.argv) > 1 else ['echo test']
    out = run_qemu(cmds, timeout=10)
    print(out)

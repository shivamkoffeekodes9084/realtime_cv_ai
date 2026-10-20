import os
import subprocess
import socket
import uvicorn

def get_local_ips():
    ips = []
    try:
        hostname = socket.gethostname()
        for ip in socket.gethostbyname_ex(hostname)[2]:
            if not ip.startswith("127."):
                ips.append(ip)
    except Exception:
        pass
    return ips

def setup_adb_reverse():
    try:
        result = subprocess.run(
            ["adb", "reverse", "tcp:8000", "tcp:8000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3,
        )
        if result.returncode == 0:
            print("  ✅ ADB Reverse Active: Phone can use http://127.0.0.1:8000 directly!")
        else:
            print(f"  ℹ️ ADB Reverse Notice: {result.stderr.strip() or 'No USB device detected'}")
    except FileNotFoundError:
        print("  ℹ️ 'adb' command not found in PATH (skip if using Wi-Fi IP).")
    except Exception as e:
        print(f"  ℹ️ ADB setup notice: {e}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)

    print("\n" + "=" * 65)
    print("  🛡️  SafeSite AI - Real-Time Vision & Tracking Backend  🛡️")
    print("=" * 65)

    setup_adb_reverse()

    local_ips = get_local_ips()
    print("\n  🌐 Available Server URLs for your Mobile App:")
    print("     • If using ADB / USB:     http://127.0.0.1:8000")
    for ip in local_ips:
        print(f"     • If using Wi-Fi / Hotspot: http://{ip}:8000")
    print(f"     • Swagger API Docs:        http://localhost:8000/docs")
    print("=" * 65 + "\n")

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

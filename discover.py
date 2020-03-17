import socket
from pprint import pprint

_PORT = 5432
_MESSAGE = b"find"


def discover_devices():
    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)

    # enable port reusage so we will be able to run multiple clients and servers on single (host, port) 
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    # enable broadcasting mode
    server.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    # bind to port
    server.bind(("", _PORT))

    # send broadcast message
    server.sendto(_MESSAGE, ("<broadcast>", _PORT))

    def get_response(timeout=0.5):   
        server.settimeout(timeout)
        while True:
            try:
                _, (ip, port) = server.recvfrom(len(_MESSAGE))
                yield ip
            except socket.timeout:
                break

    return sorted(get_response())


if __name__ == "__main__":
    pprint(discover_devices())

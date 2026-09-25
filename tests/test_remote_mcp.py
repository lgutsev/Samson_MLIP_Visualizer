import base64
import io
import json

from test_remote_client import bridge  # noqa: F401 - pytest fixture

from samson_mlip_visualizer.remote.client import SamsonClient, main
from samson_mlip_visualizer.remote.mcp_server import McpServer


def rpc(server, method, request_id=1, **params):
    return server.handle({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})


def test_initialize_and_list_tools():
    server = McpServer(client_factory=lambda: None)
    init = rpc(server, "initialize", protocolVersion="2025-06-18")["result"]
    assert init["protocolVersion"] == "2025-06-18"
    assert init["capabilities"] == {"tools": {"listChanged": False}}
    assert server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    names = {tool["name"] for tool in rpc(server, "tools/list")["result"]["tools"]}
    assert {"samson_get_structure", "samson_view", "samson_start_job", "samson_exec"} <= names
    assert rpc(server, "no/such")["error"]["code"] == -32601
    assert rpc(server, "tools/call", name="nope")["error"]["code"] == -32602


def test_tools_drive_the_bridge(bridge):  # noqa: F811
    samson, _ = bridge
    server = McpServer(client_factory=SamsonClient.from_connection_file)

    def tool(name, **arguments):
        return rpc(server, "tools/call", name=name, arguments=arguments)["result"]

    structure = json.loads(tool("samson_get_structure", models="all")["content"][0]["text"])
    assert structure["symbols"] == ["O", "H", "H", "O", "H", "H"]
    tool("samson_select", nsl="node.type atom")
    assert samson.selections == ["node.type atom"]
    tool("samson_add_atoms", atoms=[{"symbol": "C", "position": [9, 9, 9]}])
    assert samson.models[0].atoms[-1].elementSymbol == "C"

    image = tool("samson_view", width=320, height=200)["content"][0]
    assert image["type"] == "image" and image["mimeType"] == "image/png"
    assert base64.b64decode(image["data"]).startswith(b"\x89PNG")

    failure = tool("samson_delete_atoms", atoms=[99])
    assert failure["isError"] is True and "indices" in failure["content"][0]["text"]


def test_stdio_loop_writes_only_json():
    server = McpServer(client_factory=lambda: None)
    stdin = io.StringIO(
        '{"jsonrpc":"2.0","id":1,"method":"ping"}\n'
        "\n"
        '{"jsonrpc":"2.0","method":"notifications/initialized"}\n'
        "not json\n"
    )
    stdout = io.StringIO()
    server.serve(stdin, stdout)
    replies = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert replies[0] == {"jsonrpc": "2.0", "id": 1, "result": {}}
    assert replies[1]["error"]["code"] == -32700
    assert len(replies) == 2


def test_cli_capture_nsl_and_job_options(bridge, tmp_path, capsys):  # noqa: F811
    samson, _ = bridge
    assert main(["capture", str(tmp_path / "v.png"), "--width", "300", "--height", "200"]) == 0
    assert samson.captures[-1][:2] == (300, 200)
    assert main(["nsl", "node.type atom and atom.symbol O"]) == 0
    capsys.readouterr()
    # No model file in the stand-in, so the job fails at once; options still parse.
    assert main(["job", "start", "relax", "--set", "fmax=0.02", "--set", "optimizer=LBFGS"]) == 1
    assert "model" in capsys.readouterr().err


import asyncio
import json
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
        sys.stdin.reconfigure(encoding="utf-8")
    except Exception:
        pass

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

PROJECT_DIR = Path(__file__).resolve().parent
SERVER_FILE = PROJECT_DIR / "mcp_server" / "server.py"


def get_schema(tool):
    """Lấy input schema do MCP server cung cấp."""
    schema = getattr(tool, "inputSchema", None) or {}
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    return properties, required


def get_type_name(schema):
    """Đọc kiểu dữ liệu từ JSON Schema."""
    value = schema.get("type", "string")

    # Một số schema có thể khai báo nhiều kiểu, ví dụ ["string", "null"].
    if isinstance(value, list):
        value = next((item for item in value if item != "null"), "string")

    return value


def convert_value(raw, schema):
    """Chuyển chuỗi người dùng nhập thành kiểu dữ liệu tương ứng."""
    value_type = get_type_name(schema)

    if value_type == "integer":
        return int(raw)
    if value_type == "number":
        return float(raw)
    if value_type == "boolean":
        normalized = raw.strip().lower()
        if normalized in ("true", "1", "yes", "y"):
            return True
        if normalized in ("false", "0", "no", "n"):
            return False
        raise ValueError("Nhập true hoặc false.")
    if value_type in ("object", "array"):
        value = json.loads(raw)
        if value_type == "object" and not isinstance(value, dict):
            raise ValueError("Tham số này cần JSON object.")
        if value_type == "array" and not isinstance(value, list):
            raise ValueError("Tham số này cần JSON array.")
        return value

    return raw


def display_tools(tools):
    """Hiển thị danh sách tool được khám phá từ server."""
    print("\n========== MCP TOOLS ==========")

    for index, tool in enumerate(tools, start=1):
        properties, required = get_schema(tool)
        required_names = set(required)

        print(f"\n{index}. {tool.name}")
        print(f"   Mô tả: {getattr(tool, 'description', None) or '(chưa có mô tả)'}")

        if not properties:
            print("   Tham số: không có")
            continue

        for name, schema in properties.items():
            marker = "bắt buộc" if name in required_names else "tùy chọn"
            print(
                f"   - {name}: {get_type_name(schema)} "
                f"({marker})"
            )


def prompt_arguments(tool):
    """Hỏi lần lượt từng tham số dựa trên schema của tool."""
    properties, required = get_schema(tool)
    arguments = {}

    print(f"\n========== GỌI TOOL: {tool.name} ==========")
    print("Nhấn Enter để bỏ qua tham số tùy chọn.")
    print("Nhập :json để chuyển sang nhập toàn bộ arguments bằng JSON.")

    for name, schema in properties.items():
        description = schema.get("description", "")
        value_type = get_type_name(schema)
        is_required = name in required

        choices = schema.get("enum")
        if choices:
            print(f"Giá trị hợp lệ: {choices}")

        if description:
            print(f"\n{name} — {description}")
        else:
            print(f"\n{name} ({value_type})")

        if is_required:
            print("Đây là tham số bắt buộc.")

        while True:
            raw = input(f"{name}: ").strip()

            if raw == ":json":
                raw_json = input("Nhập arguments JSON: ").strip()
                try:
                    value = json.loads(raw_json)
                    if not isinstance(value, dict):
                        raise ValueError("Arguments phải là JSON object.")
                    return value
                except (json.JSONDecodeError, ValueError) as exc:
                    print(f"JSON không hợp lệ: {exc}")
                    continue

            if not raw:
                if is_required:
                    print("Không được bỏ trống tham số bắt buộc.")
                    continue

                # Không truyền tham số tùy chọn nếu người dùng bỏ qua.
                break

            try:
                value = convert_value(raw, schema)

                if choices and value not in choices:
                    print(f"Giá trị không hợp lệ. Chọn một trong: {choices}")
                    continue

                arguments[name] = value
                break

            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                print(f"Dữ liệu không hợp lệ: {exc}")

    return arguments


def display_result(result):
    """Hiển thị kết quả trả về từ MCP tool."""
    print("\n========== KẾT QUẢ ==========")

    if getattr(result, "isError", False):
        print("Tool trả về lỗi.")

    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        print(json.dumps(structured, ensure_ascii=False, indent=2))
        return

    content = getattr(result, "content", None)
    if content:
        for item in content:
            text = getattr(item, "text", None)
            print(text if text is not None else item)
    else:
        print(result)


async def main():
    if not SERVER_FILE.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy MCP server: {SERVER_FILE}"
        )

    server = StdioServerParameters(
        command=sys.executable,
        args=["-u", str(SERVER_FILE)],
        cwd=str(PROJECT_DIR),
    )

    print("Đang kết nối MCP server...")

    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as client:
            await client.initialize()
            response = await client.list_tools()
            tools = response.tools

            if not tools:
                print("Server chưa cung cấp tool nào.")
                return

            display_tools(tools)

            while True:
                print("\n========== MENU ==========")
                print("Nhập số thứ tự để chọn tool.")
                print("Nhập list để xem lại danh sách.")
                print("Nhập q để thoát.")

                choice = input("\nLựa chọn: ").strip().lower()

                if choice == "q":
                    print("Đã thoát MCP client.")
                    break

                if choice == "list":
                    display_tools(tools)
                    continue

                if not choice.isdigit():
                    print("Vui lòng nhập số thứ tự, list hoặc q.")
                    continue

                index = int(choice) - 1

                if index < 0 or index >= len(tools):
                    print("Số thứ tự không tồn tại.")
                    continue

                tool = tools[index]

                try:
                    arguments = prompt_arguments(tool)
                    print("\nArguments sẽ gửi:")
                    print(json.dumps(arguments, ensure_ascii=False, indent=2))

                    confirm = input("Gọi tool? (y/n): ").strip().lower()
                    if confirm not in ("y", "yes"):
                        print("Đã hủy.")
                        continue

                    result = await client.call_tool(
                        tool.name,
                        arguments=arguments,
                    )
                    display_result(result)

                except Exception as exc:
                    print(f"Lỗi khi gọi tool: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    asyncio.run(main())

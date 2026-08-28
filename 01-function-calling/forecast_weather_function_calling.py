"""Minh hoạ FUNCTION CALLING dự báo thời tiết (Weather Forecast) với Google Gemini SDK.

Tool `get_forecast` được định nghĩa schema thủ công VÀ thực thi ngay trong file.
Model quyết định gọi tool nào và truyền tham số (city, days); app thực thi và trả lại kết quả.

Cách chạy:
    pip install -r ../requirements.txt
    set GEMINI_API_KEY=your_api_key_here
    python forecast_weather_function_calling.py
"""

import json
from google import genai
from google.genai import types

# Khởi tạo gemini client. Sẽ tự động lấy API key từ biến môi trường GEMINI_API_KEY
client = genai.Client()

MODEL = "gemini-2.5-flash"

SYSTEM_INSTRUCTION = (
    "Bạn là trợ lý thời tiết thân thiện, trả lời người dùng bằng tiếng Việt tự nhiên. "
    "Dùng emoji phù hợp (🌧️ 🌤️ 💨 💧 ☀️). "
    "Tóm tắt thông tin dự báo thời tiết theo từng ngày một cách ngắn gọn, rõ ràng, "
    "và đưa ra lời khuyên thực tế (dặn mang ô/áo mưa nếu mưa, mặc đồ mỏng/kem chống nắng nếu nắng nóng, ...)."
)

# 1. Định nghĩa schema của tool get_forecast bằng cách thủ công
get_forecast_declaration = types.FunctionDeclaration(
    name="get_forecast",
    description="Lấy thông tin dự báo thời tiết trong vài ngày tới của một thành phố",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "city": types.Schema(
                type=types.Type.STRING, description="Tên thành phố (ví dụ: Hà Nội, Hồ Chí Minh, Đà Nẵng)"
            ),
            "days": types.Schema(
                type=types.Type.INTEGER, description="Số ngày cần dự báo (tối thiểu là 1, tối đa là 3, mặc định là 3)"
            )
        },
        required=["city"],
    ),
)

TOOLS = [types.Tool(function_declarations=[get_forecast_declaration])]


# 2. Định nghĩa hàm Python thực tế sẽ chạy khi model yêu cầu
def get_forecast(city: str, days: int = 3) -> str:
    """Trả về dữ liệu dự báo thời tiết (mock) của *city* trong *days* ngày."""
    mock_forecast_data = {
        "Hà Nội": [
            {"ngày": "2026-08-28", "nhiệt_độ_cao_nhất": "32°C", "nhiệt_độ_thấp_nhất": "26°C", "thời_tiết": "mưa rào rải rác", "khả_năng_mưa": "70%"},
            {"ngày": "2026-08-29", "nhiệt_độ_cao_nhất": "33°C", "nhiệt_độ_thấp_nhất": "27°C", "thời_tiết": "nhiều mây, có lúc hửng nắng", "khả_năng_mưa": "30%"},
            {"ngày": "2026-08-30", "nhiệt_độ_cao_nhất": "34°C", "nhiệt_độ_thấp_nhất": "27°C", "thời_tiết": "nắng nóng, trời quang", "khả_năng_mưa": "10%"},
        ],
        "Hồ Chí Minh": [
            {"ngày": "2026-08-28", "nhiệt_độ_cao_nhất": "33°C", "nhiệt_độ_thấp_nhất": "25°C", "thời_tiết": "chiều tối có mưa giông", "khả_năng_mưa": "80%"},
            {"ngày": "2026-08-29", "nhiệt_độ_cao_nhất": "32°C", "nhiệt_độ_thấp_nhất": "25°C", "thời_tiết": "mưa rào cả ngày", "khả_năng_mưa": "90%"},
            {"ngày": "2026-08-30", "nhiệt_độ_cao_nhất": "34°C", "nhiệt_độ_thấp_nhất": "26°C", "thời_tiết": "nắng gián đoạn", "khả_năng_mưa": "40%"},
        ],
        "Đà Nẵng": [
            {"ngày": "2026-08-28", "nhiệt_độ_cao_nhất": "31°C", "nhiệt_độ_thấp_nhất": "26°C", "thời_tiết": "nhiều mây", "khả_năng_mưa": "20%"},
            {"ngày": "2026-08-29", "nhiệt_độ_cao_nhất": "32°C", "nhiệt_độ_thấp_nhất": "26°C", "thời_tiết": "khô ráo, hửng nắng nhẹ", "khả_năng_mưa": "15%"},
            {"ngày": "2026-08-30", "nhiệt_độ_cao_nhất": "33°C", "nhiệt_độ_thấp_nhất": "27°C", "thời_tiết": "nắng đẹp", "khả_năng_mưa": "5%"},
        ]
    }

    # Chuẩn hoá tên thành phố đầu vào
    matched_city = None
    for k in mock_forecast_data.keys():
        if city.lower() in k.lower() or k.lower() in city.lower():
            matched_city = k
            break

    if not matched_city:
        return json.dumps({
            "error": f"Không tìm thấy dữ liệu dự báo cho thành phố '{city}'. Chỉ hỗ trợ Hà Nội, Hồ Chí Minh, Đà Nẵng."
        }, ensure_ascii=False)

    # Lấy số ngày yêu cầu (giới hạn từ 1 đến 3)
    limit_days = max(1, min(int(days), 3))
    forecast_list = mock_forecast_data[matched_city][:limit_days]

    return json.dumps({
        "city": matched_city,
        "days_forecasted": limit_days,
        "forecast": forecast_list
    }, ensure_ascii=False)


def run(prompt: str) -> str:
    """Gửi prompt tới Gemini, tự động xử lý tool calling và trả về câu trả lời."""
    contents: list[types.Content] = [
        types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
    ]

    # 3. Gửi prompt + đăng ký tools cho Gemini
    resp = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            tools=TOOLS,
            system_instruction=SYSTEM_INSTRUCTION,
        ),
    )

    # 4. Vòng lặp: Nếu Gemini gửi yêu cầu gọi tool, app thực thi và chuyển kết quả lại cho model
    while resp.function_calls:
        # Thêm câu trả lời chứa yêu cầu gọi tool của model vào danh sách hội thoại
        contents.append(resp.candidates[0].content)

        function_responses = []
        for fc in resp.function_calls:
            print(f"  [gemini yêu cầu tool] {fc.name}({fc.args})")
            
            # Thực thi hàm tương ứng
            if fc.name == "get_forecast":
                # Lấy arguments gửi từ model
                args = fc.args
                result = get_forecast(**args)
            else:
                result = f"Error: Tool {fc.name} không tồn tại."
                
            print(f"  [app thực thi và trả về] -> {result}")
            
            # Đóng gói kết quả để gửi lại cho model
            function_responses.append(
                types.Part.from_function_response(
                    name=fc.name, response={"result": result}
                )
            )

        # Thêm kết quả chạy tool vào nội dung hội thoại tiếp theo
        contents.append(types.Content(role="user", parts=function_responses))
        
        # Gọi tiếp model với thông tin cập nhật
        resp = client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                tools=TOOLS,
                system_instruction=SYSTEM_INSTRUCTION,
            ),
        )

    # 5. Trả về kết quả tổng hợp cuối cùng sau khi đã xử lý hết các tool calls
    return resp.text


if __name__ == "__main__":
    import os
    # Kiểm tra xem API key đã được set chưa
    if not os.getenv("GEMINI_API_KEY"):
        print("⚠️ CẢNH BÁO: Bạn chưa set biến môi trường GEMINI_API_KEY.")
        print("Vui lòng chạy lệnh: set GEMINI_API_KEY=your_key")
        print("Hoặc nhập trực tiếp tại đây để thử nghiệm:")
        key = input("GEMINI_API_KEY: ").strip()
        if key:
            os.environ["GEMINI_API_KEY"] = key
        else:
            print("Không có API key, ứng dụng có thể gặp lỗi.")

    question = "Thời tiết Đà Nẵng và Hà Nội trong 2 ngày tới sẽ như thế nào?"
    print(f"\nUser: {question}\n")
    try:
        answer = run(question)
        print("\nTrả lời từ Gemini:")
        print(answer)
    except Exception as e:
        print(f"\nCó lỗi xảy ra: {e}")

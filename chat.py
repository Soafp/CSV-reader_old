from tools import tools, dispatch
import json, os, streamlit as st
from dotenv import load_dotenv
from openai import OpenAI 

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/v1",
)
MODEL = "openai/gpt-oss-120b"

CHART_TOOLS = {"create_chart", "resample_timeseries", "compare_months", "compare_columns"}


CHART_KEYWORDS = {"chart", "plot", "graph", "line", "bar", "scatter", "pie", "histogram", "area", "trend", "visuali"}

MAX_TOOL_ITERATIONS = 5

def get_relevant_tools(user_input):
    if any(w in user_input.lower() for w in CHART_KEYWORDS):
        return tools
    return [t for t in tools if t["function"]["name"] not in CHART_TOOLS]


def init_chat(df):
    col_lines = [f"  {col} ({dtype})" for col, dtype in df.dtypes.items()]
    col_info = "\n".join(col_lines[:30])

    system_prompt = f"""You are a data analyst. CSV: {df.shape[0]} rows × {df.shape[1]} cols.

Columns:
{col_info}

RULES:
- When the user asks for a pie chart about a specific row (e.g. a country, person, product):
  1. Filter to that single row
  2. Use the NUMERIC columns as pie slices (column name = label, value = slice size)
  3. Never plot all rows of a single column as a pie chart unless asked to compare
- When the user asks for a pie chart comparing multiple rows (e.g. "top 5 countries by GDP"):
    -Use the relevant column as slice values, row identifiers as labels
- NEVER respond with a table or text description of chart data. ALWAYS call the create_chart tool.
- If the user says "pie chart", "bar chart", "line chart" etc., you MUST call create_chart — no exceptions.
- Do not ask the user if they want a chart. Just make it.
- When the user says "first N", use sort_data or query_data with limit=N to get those rows first, then pass the result to create_chart
- For scatter/bar charts of multiple countries, use x_col = the country/name column, y_col = the value column
"""

    st.session_state.messages = [{"role": "system", "content": system_prompt}]


def clean_messages(messages):
    cleaned = []
    for msg in messages:
        if isinstance(msg, dict):
            cleaned.append({k: v for k, v in msg.items() if k != "reasoning"})
        else:
            # Convert ChatCompletionMessage object to dict
            d = {"role": msg.role, "content": msg.content}
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                d["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                    }
                    for tc in msg.tool_calls
                ]
            cleaned.append(d)   
    return cleaned


def chat(user_input, df):
    if "chart_counter" not in st.session_state:
        st.session_state.chart_counter = 0

    active_tools = get_relevant_tools(user_input)

    history = st.session_state.chat_history[-2:] if len(st.session_state.chat_history) >= 2 else []
    last_exchange = [{"role": r, "content": t} for r, t in history]

    messages = [
        st.session_state.messages[0],
        *last_exchange,
        {"role": "user", "content": user_input}
    ]

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=clean_messages(messages),
            tools=active_tools,
            tool_choice="auto"
        )
    except Exception as e:
        err = str(e)
        if "tool call validation failed" in err or "was not in request.tools" in err:
            response = client.chat.completions.create(
                model=MODEL,
                messages=clean_messages(messages),
            )
        else:
            return f"Error: {err}"
        
    iteration = 0
    while response.choices[0].finish_reason == "tool_calls" and iteration < MAX_TOOL_ITERATIONS:
        iteration += 1  
        tool_calls = response.choices[0].message.tool_calls
        messages.append(response.choices[0].message)

        for tool_call in tool_calls:
            fn_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            if fn_name in CHART_TOOLS:
                fig, result = dispatch(fn_name, args, df)
                if fig:
                    cols = args.get("value_cols") or [args.get("value_col", args.get("y_col", args.get("x_col", "")))]
                    title = " vs ".join(cols) if isinstance(cols, list) else cols
                    st.session_state.chart_counter += 1
                    st.session_state.figs.append((fig, title, st.session_state.chart_counter))
            else:
                result = dispatch(fn_name, args, df)

            tool_result_str = json.dumps(result)
            if len(tool_result_str) > 1000:
                tool_result_str = tool_result_str[:1000] + "... [truncated]"

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result_str
            })

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=clean_messages(messages),
                tools=active_tools,
                tool_choice="auto"
            )
        except Exception as e:
            err = str(e)
            if "tool call validation failed" in err or "was not in request.tools" in err:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=clean_messages(messages),
                )
            else:
                return f"Error: {err}"

    reply = response.choices[0].message.content
    return reply


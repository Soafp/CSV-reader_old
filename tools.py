import pandas as pd
import plotly.express as px

CHART_COLORS = ["#EF4444", "#3B82F6", "#EAB308",  "#10B981", "#8B5CF6", "#F97316"]


def read_csv(file_path):
    df = pd.read_csv(file_path, sep=None, engine="python")
    return df


def get_dataframe_info(df):
    return {
        "rows": len(df),
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }


def get_detailed_info(df):
    return {
        "null_counts": df.isnull().sum().to_dict(),
        "sample": df.head(2).to_dict(orient="records"),
        "describe": df.describe().round(2).to_dict()
    }


def query_data(df, filters=None, columns=None, limit=20):
    result = df.copy()
    if filters:
        for col, op, val in filters:
            if col not in result.columns:
                return {"error": f"Column '{col}' not found"}
            if op == "==":         result = result[result[col] == val]
            elif op == ">":        result = result[result[col] > val]
            elif op == "<":        result = result[result[col] < val]
            elif op == ">=":       result = result[result[col] >= val]
            elif op == "<=":       result = result[result[col] <= val]
            elif op == "contains": result = result[result[col].astype(str).str.contains(val, case=False)]
    if columns:
        columns = [c for c in columns if c in result.columns]
        result = result[columns]
    return result.head(limit).to_dict(orient="records")


def group_and_aggregate(df, group_col, agg_col, agg_func="sum"):
    if group_col not in df.columns:
        return {"error": f"Column '{group_col}' not found"}
    if agg_col not in df.columns:
        return {"error": f"Column '{agg_col}' not found"}
    funcs = {"sum": "sum", "mean": "mean", "count": "count", "min": "min", "max": "max"}
    if agg_func not in funcs:
        return {"error": f"Unknown function '{agg_func}'. Use: {list(funcs.keys())}"}
    result = df.groupby(group_col)[agg_col].agg(agg_func).reset_index()
    result.columns = [group_col, f"{agg_func}_{agg_col}"]
    return result.to_dict(orient="records")


def sort_data(df, sort_col, ascending=False, limit=10):
    if sort_col not in df.columns:
        return {"error": f"Column '{sort_col}' not found"}
    result = df.sort_values(sort_col, ascending=ascending).head(limit)
    return result.to_dict(orient="records")


def extract_date_part(df, date_col, part="month"):
    if date_col not in df.columns:
        return {"error": f"Column '{date_col}' not found"}
    try:
        parsed = pd.to_datetime(df[date_col], dayfirst=True)
    except Exception as e:
        return {"error": f"Could not parse dates: {str(e)}"}
    parts = {
        "year":    parsed.dt.year,
        "month":   parsed.dt.month,
        "day":     parsed.dt.day,
        "weekday": parsed.dt.day_name(),
        "quarter": parsed.dt.quarter,
        "week":    parsed.dt.isocalendar().week.astype(int)
    }
    if part not in parts:
        return {"error": f"Unknown part '{part}'. Use: {list(parts.keys())}"}
    temp = df.copy()
    temp["__part__"] = parts[part]
    return temp.groupby("__part__").size().reset_index(name="count").to_dict(orient="records")


def compute_stat(df, column, stat="mean", filters=None):
    if column not in df.columns:
        return {"error": f"Column '{column}' not found"}
    result = df.copy()
    if filters:
        for col, op, val in filters:
            if col not in result.columns:
                return {"error": f"Filter column '{col}' not found"}
            if op == "==":         result = result[result[col] == val]
            elif op == ">":        result = result[result[col] > val]
            elif op == "<":        result = result[result[col] < val]
            elif op == ">=":       result = result[result[col] >= val]
            elif op == "<=":       result = result[result[col] <= val]
            elif op == "contains": result = result[result[col].astype(str).str.contains(val, case=False)]
    col = result[column]
    stats = {
        "mean": col.mean(), "sum": col.sum(), "min": col.min(),
        "max": col.max(), "median": col.median(), "std": col.std(),
        "count": col.count()
    }
    if stat not in stats:
        return {"error": f"Unknown stat '{stat}'. Use: {list(stats.keys())}"}
    return {"column": column, "stat": stat, "value": round(stats[stat], 4), "rows_used": len(result)}


def correlation(df, columns=None):
    numeric = df.select_dtypes(include="number")
    if columns:
        columns = [c for c in columns if c in numeric.columns]
        numeric = numeric[columns]
    if numeric.shape[1] < 2:
        return {"error": "Need at least 2 numeric columns for correlation"}
    return numeric.corr().round(3).to_dict()


def value_counts(df, column, limit=20):
    if column not in df.columns:
        return {"error": f"Column '{column}' not found"}
    result = df[column].value_counts().head(limit).reset_index()
    result.columns = [column, "count"]
    return result.to_dict(orient="records")


def detect_outliers(df, column):
    if column not in df.columns:
        return {"error": f"Column '{column}' not found"}
    if not pd.api.types.is_numeric_dtype(df[column]):
        return {"error": f"Column '{column}' is not numeric"}
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    mask = (df[column] < q1 - 1.5 * iqr) | (df[column] > q3 + 1.5 * iqr)
    outliers = df[mask][[column]]
    return {
        "column": column,
        "outlier_count": len(outliers),
        "bounds": {"lower": round(q1 - 1.5 * iqr, 4), "upper": round(q3 + 1.5 * iqr, 4)},
        "outliers": outliers.to_dict(orient="records")
    }


def create_chart(df, chart_type, x_col, y_col=None, filter_col=None, filter_val=None, limit=None):
    if limit:
        df = df.head(limit)

    if filter_col and filter_val:
        row = df[df[filter_col].astype(str).str.lower() == str(filter_val).lower()]
        if row.empty:
            return None, {"error": f"No row found where {filter_col} == {filter_val}"}
        if chart_type in ("pie", "bar", "line", "scatter") and len(row) == 1:
            numeric = row.select_dtypes(include="number").iloc[0]
            numeric = numeric.drop([c for c in numeric.index if "rank" in c.lower() or "id" in c.lower()], errors="ignore")
            plot_df = numeric.reset_index()
            plot_df.columns = ["Metric", "Value"]
            if chart_type == "pie":
                fig = px.pie(plot_df, names="Metric", values="Value", color_discrete_sequence=CHART_COLORS)
            elif chart_type == "bar":
                fig = px.bar(plot_df, x="Metric", y="Value", color_discrete_sequence=CHART_COLORS)
            elif chart_type == "scatter":
                fig = px.scatter(plot_df, x="Metric", y="Value", color_discrete_sequence=CHART_COLORS)
            elif chart_type == "line":
                fig = px.line(plot_df, x="Metric", y="Value", color_discrete_sequence=CHART_COLORS)
            else:
                fig = px.bar(plot_df, x="Metric", y="Value", color_discrete_sequence=CHART_COLORS)  # fallback
            return fig, {"status": "ok", "chart_type": chart_type}

    if x_col not in df.columns:
        return None, {"error": f"Column '{x_col}' not found. Available: {df.columns.tolist()}"}
    if y_col and y_col not in df.columns:
        return None, {"error": f"Column '{y_col}' not found. Available: {df.columns.tolist()}"}
    if chart_type == "bar":
        fig = px.bar(df, x=x_col, y=y_col, color_discrete_sequence=CHART_COLORS)
    elif chart_type == "pie":
        fig = px.pie(df, names=x_col, values=y_col, color_discrete_sequence=CHART_COLORS)
    elif chart_type == "scatter":
        fig = px.scatter(df, x=x_col, y=y_col, color_discrete_sequence=CHART_COLORS)
    elif chart_type == "line":
        fig = px.line(df, x=x_col, y=y_col, color_discrete_sequence=CHART_COLORS)
    elif chart_type == "histogram":
        fig = px.histogram(df, x=x_col, color_discrete_sequence=CHART_COLORS)
    else:
        return None, {"error": f"Unknown chart type: {chart_type}"}

    return fig, {"status": "ok", "chart_type": chart_type}  

def resample_timeseries(df, date_col, value_col, freq="month", agg_func="sum"):
    if date_col not in df.columns:
        return None, {"error": f"Column '{date_col}' not found"}
    if value_col not in df.columns:
        return None, {"error": f"Column '{value_col}' not found"}
    freq_map = {"day": "D", "week": "W", "month": "ME", "year": "YE"}
    if freq not in freq_map:
        return None, {"error": f"Unknown freq '{freq}'. Use: {list(freq_map.keys())}"}
    temp = df.copy()
    temp[date_col] = pd.to_datetime(temp[date_col], dayfirst=True)
    temp = temp.set_index(date_col)
    resampled = temp[value_col].resample(freq_map[freq]).agg(agg_func).reset_index()
    resampled.columns = [date_col, value_col]
    fig = px.line(resampled, x=date_col, y=value_col,
                  title=f"{value_col} per {freq} ({agg_func})",
                  color_discrete_sequence=CHART_COLORS)
    return fig, {"status": "ok", "rows": len(resampled)}


def compare_months(df, date_col, value_col, months, chart_type="line"):
    if date_col not in df.columns:
        return None, {"error": f"Column '{date_col}' not found"}
    if value_col not in df.columns:
        return None, {"error": f"Column '{value_col}' not found"}
    temp = df.copy()
    temp[date_col] = pd.to_datetime(temp[date_col], dayfirst=True)
    temp["__month__"] = temp[date_col].dt.month
    temp["__day__"] = temp[date_col].dt.day
    filtered = temp[temp["__month__"].isin(months)].copy()
    filtered["Month"] = filtered[date_col].dt.strftime("%B")
    title = f"{value_col}: {', '.join(filtered['Month'].unique())} comparison"
    kwargs = dict(x="__day__", y=value_col, color="Month", title=title,
                  labels={"__day__": "Day of month"}, color_discrete_sequence=CHART_COLORS)

    if chart_type == "scatter":
        fig = px.scatter(filtered, **kwargs)
    elif chart_type == "bar":
        fig = px.bar(filtered, **kwargs, barmode="group")
    elif chart_type == "area":
        fig = px.area(filtered, **kwargs)
    else:
        fig = px.line(filtered, **kwargs)

    return fig, {"status": "ok"}


def compare_columns(df, date_col, value_cols, chart_type="line"):
    if date_col not in df.columns:
        return None, {"error": f"Column '{date_col}' not found"}
    missing = [c for c in value_cols if c not in df.columns]
    if missing:
        return None, {"error": f"Columns not found: {missing}. Available: {df.columns.tolist()}"}
    temp = df.copy()
    temp[date_col] = pd.to_datetime(temp[date_col], dayfirst=True, errors="ignore")
    temp = temp.sort_values(date_col)
    melted = temp[[date_col] + value_cols].melt(id_vars=date_col, var_name="Series", value_name="Value")
    title = " vs ".join(value_cols)

    if chart_type == "line":
        fig = px.line(melted, x=date_col, y="Value", color="Series", title=title, color_discrete_sequence=CHART_COLORS)
    elif chart_type == "bar":
        fig = px.bar(melted, x=date_col, y="Value", color="Series", title=title, barmode="group", color_discrete_sequence=CHART_COLORS)
    elif chart_type == "scatter":
        fig = px.scatter(melted, x=date_col, y="Value", color="Series", title=title, color_discrete_sequence=CHART_COLORS)
    elif chart_type == "area":
        fig = px.area(melted, x=date_col, y="Value", color="Series", title=title, color_discrete_sequence=CHART_COLORS)
    else:
        return None, {"error": f"Unsupported chart_type '{chart_type}'. Use: line, bar, scatter, area"}

    return fig, {"status": "ok"}


def dispatch(fn_name, args, df):
    if fn_name == "get_dataframe_info":
        return get_dataframe_info(df)
    elif fn_name == "get_detailed_info":
        return get_detailed_info(df)
    elif fn_name == "query_data":
        return query_data(df, args.get("filters"), args.get("columns"), args.get("limit", 20))
    elif fn_name == "group_and_aggregate":
        return group_and_aggregate(df, args["group_col"], args["agg_col"], args.get("agg_func", "sum"))
    elif fn_name == "sort_data":
        return sort_data(df, args["sort_col"], args.get("ascending", False), args.get("limit", 10))
    elif fn_name == "extract_date_part":
        return extract_date_part(df, args["date_col"], args.get("part", "month"))
    elif fn_name == "compute_stat":
        return compute_stat(df, args["column"], args.get("stat", "mean"), args.get("filters"))
    elif fn_name == "correlation":
        return correlation(df, args.get("columns"))
    elif fn_name == "value_counts":
        return value_counts(df, args["column"], args.get("limit", 20))
    elif fn_name == "detect_outliers":
        return detect_outliers(df, args["column"])
    elif fn_name == "create_chart":
        # Normalize chart type synonyms
        chart_type = args["chart_type"]
        bar_synonyms = ["bardiagram", "bar diagram", "bar_diagram", "barchart"]
        if chart_type.lower().replace(" ", "") in [s.replace(" ", "") for s in bar_synonyms]:
            args["chart_type"] = "bar"
        return create_chart(df, args["chart_type"], args["x_col"], args.get("y_col"),
                            args.get("filter_col"), args.get("filter_val"), args.get("limit"))
    elif fn_name == "resample_timeseries":
        return resample_timeseries(df, args["date_col"], args["value_col"], args.get("freq", "month"), args.get("agg_func", "sum"))
    elif fn_name == "compare_months":
        return compare_months(df, args["date_col"], args["value_col"], args["months"], args.get("chart_type", "line"))
    elif fn_name == "compare_columns":
        return compare_columns(df, args["date_col"], args["value_cols"], args.get("chart_type", "line"))
    else:
        return {"error": f"Unknown tool: {fn_name}"}


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_dataframe_info",
            "description": "Row count, columns, dtypes.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_detailed_info",
            "description": "Null counts, sample rows, stats.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_data",
            "description": "Filter and retrieve rows.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filters": {"type": "array", "description": "[col, op, val] list. Ops: ==,>,<,>=,<=,contains.", "items": {"type": "array"}},
                    "columns": {"type": "array", "description": "Columns to return.", "items": {"type": "string"}},
                    "limit":   {"type": "integer", "description": "Max rows. Default 20."}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "group_and_aggregate",
            "description": "Group by column, aggregate another.",
            "parameters": {
                "type": "object",
                "properties": {
                    "group_col": {"type": "string"},
                    "agg_col":   {"type": "string"},
                    "agg_func":  {"type": "string", "enum": ["sum", "mean", "count", "min", "max"]}
                },
                "required": ["group_col", "agg_col"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "sort_data",
            "description": "Top/bottom N rows by column.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sort_col":  {"type": "string"},
                    "ascending": {"type": "boolean"},
                    "limit":     {"type": "integer"}
                },
                "required": ["sort_col"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_date_part",
            "description": "Count rows by date part.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date_col": {"type": "string"},
                    "part":     {"type": "string", "enum": ["year", "month", "day", "weekday", "quarter", "week"]}
                },
                "required": ["date_col"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compute_stat",
            "description": "Compute stat on a column with optional filters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column":  {"type": "string"},
                    "stat":    {"type": "string", "enum": ["mean", "sum", "min", "max", "median", "std", "count"]},
                    "filters": {"type": "array", "description": "[col, op, val] list.", "items": {"type": "array"}}
                },
                "required": ["column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "correlation",
            "description": "Correlation matrix of numeric columns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "columns": {"type": "array", "items": {"type": "string"}}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "value_counts",
            "description": "Frequency count of a column's values.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "limit":  {"type": "integer"}
                },
                "required": ["column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "detect_outliers",
            "description": "IQR outlier detection on a column.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"}
                },
                "required": ["column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_chart",
            "description": "Single-series chart. For a pie chart of one row's columns (e.g. one country), pass filter_col and filter_val to select that row — columns become slices.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chart_type":  {"type": "string", "enum": ["bar", "pie", "scatter", "line", "histogram"]},
                    "x_col":       {"type": "string"},
                    "y_col":       {"type": "string"},
                    "filter_col":  {"type": "string", "description": "Column to filter on (e.g. 'Country or region')"},
                    "filter_val":  {"type": "string", "description": "Value to match (e.g. 'Denmark')"},
                    "limit":       {"type": "integer", "description": "Max number of rows to plot. Use when user says 'first N' or 'top N'."}
                },
                "required": ["chart_type", "x_col"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "resample_timeseries",
            "description": "Aggregate one column over time, plot as line.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date_col":  {"type": "string"},
                    "value_col": {"type": "string"},
                    "freq":      {"type": "string", "enum": ["day", "week", "month", "year"]},
                    "agg_func":  {"type": "string", "enum": ["sum", "mean", "min", "max"]}
                },
                "required": ["date_col", "value_col"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_months",
            "description": "Plot months as separate series, e.g. Jan vs Feb.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date_col":   {"type": "string"},
                    "value_col":  {"type": "string"},
                    "months":     {"type": "array", "description": "Month numbers e.g. [1,2].", "items": {"type": "integer"}},
                    "chart_type": {"type": "string", "enum": ["line", "bar", "scatter", "area"]}
                },
                "required": ["date_col", "value_col", "months"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_columns",
            "description": "Plot multiple columns on same chart, e.g. price vs consumption.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date_col":   {"type": "string"},
                    "value_cols": {"type": "array", "items": {"type": "string"}},
                    "chart_type": {"type": "string", "enum": ["line", "bar", "scatter", "area"]}
                },
                "required": ["date_col", "value_cols"]
            }
        }
    }
]
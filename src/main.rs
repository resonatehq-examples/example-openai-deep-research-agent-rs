use resonate::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value as JsonValue};

/// One message in the running OpenAI conversation.
#[derive(Serialize, Deserialize, Debug, Clone)]
struct Message {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

/// A single tool call returned by the model.
#[derive(Serialize, Deserialize, Debug, Clone)]
struct ToolCall {
    id: String,
    #[serde(rename = "type")]
    kind: String,
    function: ToolCallFunction,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct ToolCallFunction {
    name: String,
    arguments: String,
}

/// Args passed to the `prompt` leaf — the running message log plus a
/// tool-access flag. Tool access is gated by depth: once depth reaches 0
/// the model must summarize directly instead of recursing.
#[derive(Serialize, Deserialize, Debug, Clone)]
struct PromptArgs {
    messages: Vec<Message>,
    has_tool_access: bool,
}

const SYSTEM_PROMPT: &str = r#"
You are a recursive research agent.

When given a broad or high-level topic, break it down into 2-3 semantically meaningful subtopics and call the "research" tool for each one individually.

Do not call the research tool if the topic is already well understood or deeply specific. Instead, summarize the topic directly instead of calling the tool.

Always respond with either:
1. A summary paragraph of the topic, or
2. One or more tool calls, each with a single subtopic to be researched.

Be concise and respond in plain English. Avoid repeating the topic verbatim in the subtopics.
"#;

/// Recursive research workflow.
///
/// Given a topic, asks OpenAI to break it into subtopics and recursively
/// dispatches a fresh `research` invocation for each subtopic. All sibling
/// subtopics are spawned in parallel via `ctx.run(research, ...).spawn()`,
/// then collected. Each level of recursion is durable — if the worker
/// crashes mid-flight, completed subtopic results are replayed from the
/// log instead of being re-asked of the model.
#[resonate::function]
async fn research(ctx: &Context, topic: String, depth: i32) -> Result<String> {
    let mut messages = vec![
        Message {
            role: "system".into(),
            content: Some(SYSTEM_PROMPT.into()),
            tool_calls: None,
            tool_call_id: None,
        },
        Message {
            role: "user".into(),
            content: Some(format!("Research {}", topic)),
            tool_calls: None,
            tool_call_id: None,
        },
    ];

    loop {
        // Ask the LLM. Tool access is allowed only while depth > 0.
        let message: Message = ctx
            .run(
                prompt,
                PromptArgs {
                    messages: messages.clone(),
                    has_tool_access: depth > 0,
                },
            )
            .await?;

        messages.push(message.clone());

        // If the model wants to research subtopics, recursively spawn each
        // one and await their results — fan-out / fan-in.
        if let Some(tool_calls) = message.tool_calls.clone() {
            let mut handles = Vec::new();
            for tool_call in tool_calls {
                if tool_call.function.name == "research" {
                    let args: JsonValue =
                        serde_json::from_str(&tool_call.function.arguments).map_err(|e| {
                            Error::EncodingError(format!("bad tool args: {}", e))
                        })?;
                    let subtopic = args
                        .get("topic")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_string();

                    let handle = ctx
                        .run(research, (subtopic, depth - 1))
                        .spawn()
                        .await?;
                    handles.push((tool_call, handle));
                }
            }

            for (tool_call, handle) in handles {
                let result = handle.await?;
                messages.push(Message {
                    role: "tool".into(),
                    content: Some(result),
                    tool_calls: None,
                    tool_call_id: Some(tool_call.id),
                });
            }
        } else {
            return Ok(message.content.unwrap_or_default());
        }
    }
}

/// OpenAI chat-completions call. A leaf function — durable on its own,
/// so a crash mid-stream doesn't replay the API call.
#[resonate::function]
async fn prompt(args: PromptArgs) -> Result<Message> {
    let api_key = std::env::var("OPENAI_API_KEY").map_err(|_| {
        Error::EncodingError("OPENAI_API_KEY environment variable not set".into())
    })?;

    let mut body = json!({
        "model": "gpt-5",
        "messages": args.messages,
    });

    if args.has_tool_access {
        body["tools"] = json!([
            {
                "type": "function",
                "function": {
                    "name": "research",
                    "description": "Research a given topic",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "topic": {
                                "type": "string",
                                "description": "The topic to research"
                            }
                        },
                        "required": ["topic"]
                    }
                }
            }
        ]);
    }

    let client = reqwest::Client::new();
    let resp = client
        .post("https://api.openai.com/v1/chat/completions")
        .bearer_auth(api_key)
        .json(&body)
        .send()
        .await
        .map_err(|e| Error::EncodingError(format!("openai request failed: {}", e)))?;

    let status = resp.status();
    let text = resp
        .text()
        .await
        .map_err(|e| Error::EncodingError(format!("openai response read failed: {}", e)))?;

    if !status.is_success() {
        return Err(Error::EncodingError(format!(
            "openai returned {}: {}",
            status, text
        )));
    }

    let parsed: JsonValue = serde_json::from_str(&text)
        .map_err(|e| Error::EncodingError(format!("openai response parse failed: {}", e)))?;

    let message_value = parsed
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("message"))
        .ok_or_else(|| Error::EncodingError("openai response missing choices[0].message".into()))?;

    let message: Message = serde_json::from_value(message_value.clone())
        .map_err(|e| Error::EncodingError(format!("message parse failed: {}", e)))?;

    Ok(message)
}

#[tokio::main]
async fn main() {
    // CLI: research <id> <topic> [depth]
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: research <id> <topic> [depth]");
        std::process::exit(1);
    }
    let id = args[1].clone();
    let topic = args[2].clone();
    let depth: i32 = args
        .get(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);

    let resonate = Resonate::new(ResonateConfig {
        url: Some("http://localhost:8001".into()),
        ..Default::default()
    });

    resonate.register(research).unwrap();
    resonate.register(prompt).unwrap();

    println!("Starting research workflow {} on topic {:?} (depth {})", id, topic, depth);

    let result: String = resonate
        .run(&id, research, (topic, depth))
        .await
        .expect("research workflow failed");

    println!("\n--- Result ---\n{}", result);
}

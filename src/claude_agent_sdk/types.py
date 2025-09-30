"""Type definitions for Claude SDK."""

import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypedDict

from typing_extensions import NotRequired

if TYPE_CHECKING:
    from mcp.server import Server as McpServer

# Permission modes
PermissionMode = Literal["default", "acceptEdits", "plan", "bypassPermissions"]

# Agent definitions
SettingSource = Literal["user", "project", "local"]


class SystemPromptPreset(TypedDict):
    """System prompt preset configuration."""

    type: Literal["preset"]
    preset: Literal["claude_code"]
    append: NotRequired[str]


@dataclass
class AgentDefinition:
    """Agent definition configuration."""

    description: str
    prompt: str
    tools: list[str] | None = None
    model: Literal["sonnet", "opus", "haiku", "inherit"] | None = None


# Permission Update types (matching TypeScript SDK)
PermissionUpdateDestination = Literal[
    "userSettings", "projectSettings", "localSettings", "session"
]

PermissionBehavior = Literal["allow", "deny", "ask"]


@dataclass
class PermissionRuleValue:
    """Permission rule value."""

    tool_name: str
    rule_content: str | None = None


@dataclass
class PermissionUpdate:
    """Permission update configuration."""

    type: Literal[
        "addRules",
        "replaceRules",
        "removeRules",
        "setMode",
        "addDirectories",
        "removeDirectories",
    ]
    rules: list[PermissionRuleValue] | None = None
    behavior: PermissionBehavior | None = None
    mode: PermissionMode | None = None
    directories: list[str] | None = None
    destination: PermissionUpdateDestination | None = None


# Tool callback types
@dataclass
class ToolPermissionContext:
    """Context information for tool permission callbacks."""

    signal: Any | None = None  # Future: abort signal support
    suggestions: list[PermissionUpdate] = field(
        default_factory=list
    )  # Permission suggestions from CLI


# Match TypeScript's PermissionResult structure
@dataclass
class PermissionResultAllow:
    """Allow permission result."""

    behavior: Literal["allow"] = "allow"
    updated_input: dict[str, Any] | None = None
    updated_permissions: list[PermissionUpdate] | None = None


@dataclass
class PermissionResultDeny:
    """Deny permission result."""

    behavior: Literal["deny"] = "deny"
    message: str = ""
    interrupt: bool = False


PermissionResult = PermissionResultAllow | PermissionResultDeny

CanUseTool = Callable[
    [str, dict[str, Any], ToolPermissionContext], Awaitable[PermissionResult]
]


##### Hook types
# Supported hook event types. Due to setup limitations, the Python SDK does not
# support SessionStart, SessionEnd, and Notification hooks.
HookEvent = (
    Literal["PreToolUse"]
    | Literal["PostToolUse"]
    | Literal["UserPromptSubmit"]
    | Literal["Stop"]
    | Literal["SubagentStop"]
    | Literal["PreCompact"]
)


# See https://docs.anthropic.com/en/docs/claude-code/hooks#advanced%3A-json-output
# for documentation of the output types. Currently, "continue", "stopReason",
# and "suppressOutput" are not supported in the Python SDK.
class HookJSONOutput(TypedDict):
    # Whether to block the action related to the hook.
    decision: NotRequired[Literal["block"]]
    # Optionally add a system message that is not visible to Claude but saved in
    # the chat transcript.
    systemMessage: NotRequired[str]
    # See each hook's individual "Decision Control" section in the documentation
    # for guidance.
    hookSpecificOutput: NotRequired[Any]


@dataclass
class HookContext:
    """Context information for hook callbacks."""

    signal: Any | None = None  # Future: abort signal support


HookCallback = Callable[
    # HookCallback input parameters:
    # - input
    #   See https://docs.anthropic.com/en/docs/claude-code/hooks#hook-input for
    #   the type of 'input', the first value.
    # - tool_use_id
    # - context
    [dict[str, Any], str | None, HookContext],
    Awaitable[HookJSONOutput],
]


# Hook matcher configuration
@dataclass
class HookMatcher:
    """Hook matcher configuration."""

    # See https://docs.anthropic.com/en/docs/claude-code/hooks#structure for the
    # expected string value. For example, for PreToolUse, the matcher can be
    # a tool name like "Bash" or a combination of tool names like
    # "Write|MultiEdit|Edit".
    matcher: str | None = None

    # A list of Python functions with function signature HookCallback
    hooks: list[HookCallback] = field(default_factory=list)


# MCP Server config
class McpStdioServerConfig(TypedDict):
    """MCP stdio server configuration."""

    type: NotRequired[Literal["stdio"]]  # Optional for backwards compatibility
    command: str
    args: NotRequired[list[str]]
    env: NotRequired[dict[str, str]]


class McpSSEServerConfig(TypedDict):
    """MCP SSE server configuration."""

    type: Literal["sse"]
    url: str
    headers: NotRequired[dict[str, str]]


class McpHttpServerConfig(TypedDict):
    """MCP HTTP server configuration."""

    type: Literal["http"]
    url: str
    headers: NotRequired[dict[str, str]]


class McpSdkServerConfig(TypedDict):
    """SDK MCP server configuration."""

    type: Literal["sdk"]
    name: str
    instance: "McpServer"


McpServerConfig = (
    McpStdioServerConfig | McpSSEServerConfig | McpHttpServerConfig | McpSdkServerConfig
)


# Content block types
@dataclass
class TextBlock:
    """Text content block."""

    text: str


@dataclass
class ThinkingBlock:
    """Thinking content block."""

    thinking: str
    signature: str


@dataclass
class ToolUseBlock:
    """Tool use content block."""

    id: str
    name: str
    input: dict[str, Any]


@dataclass
class ToolResultBlock:
    """Tool result content block."""

    tool_use_id: str
    content: str | list[dict[str, Any]] | None = None
    is_error: bool | None = None


ContentBlock = TextBlock | ThinkingBlock | ToolUseBlock | ToolResultBlock


# Message types
@dataclass
class UserMessage:
    """User message."""

    content: str | list[ContentBlock]
    parent_tool_use_id: str | None = None


@dataclass
class AssistantMessage:
    """Assistant message with content blocks."""

    content: list[ContentBlock]
    model: str
    parent_tool_use_id: str | None = None


@dataclass
class SystemMessage:
    """System message with metadata."""

    subtype: str
    data: dict[str, Any]


@dataclass
class ResultMessage:
    """Result message with cost and usage information."""

    subtype: str
    duration_ms: int
    duration_api_ms: int
    is_error: bool
    num_turns: int
    session_id: str
    total_cost_usd: float | None = None
    usage: dict[str, Any] | None = None
    result: str | None = None


@dataclass
class StreamEvent:
    """Stream event for partial message updates during streaming."""

    uuid: str
    session_id: str
    event: dict[str, Any]  # The raw Anthropic API stream event
    parent_tool_use_id: str | None = None


Message = UserMessage | AssistantMessage | SystemMessage | ResultMessage | StreamEvent


@dataclass
class ClaudeAgentOptions:
    """Configuration options for interacting with the OpenAI GPT models."""

    # OpenAI configuration
    model: str | None = None
    openai_api_key: str | None = None
    openai_organization: str | None = None
    openai_project: str | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None
    response_format: str | dict[str, Any] | None = None
    openai_client: Any | None = None  # Dependency injection hook for testing

    # Prompt configuration
    system_prompt: str | SystemPromptPreset | None = None
    extra_headers: dict[str, str] = field(default_factory=dict)

    # Legacy Claude specific options retained for compatibility.
    # They are ignored when using the OpenAI backend but kept so existing
    # integrations don't break when upgrading the SDK.
    allowed_tools: list[str] = field(default_factory=list)
    mcp_servers: dict[str, McpServerConfig] | str | Path = field(default_factory=dict)
    permission_mode: PermissionMode | None = None
    continue_conversation: bool = False
    resume: str | None = None
    max_turns: int | None = None
    disallowed_tools: list[str] = field(default_factory=list)
    permission_prompt_tool_name: str | None = None
    cwd: str | Path | None = None
    settings: str | None = None
    add_dirs: list[str | Path] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    extra_args: dict[str, str | None] = field(default_factory=dict)
    max_buffer_size: int | None = None
    debug_stderr: Any = sys.stderr
    stderr: Callable[[str], None] | None = None
    can_use_tool: CanUseTool | None = None
    hooks: dict[HookEvent, list[HookMatcher]] | None = None
    user: str | None = None
    include_partial_messages: bool = False
    fork_session: bool = False
    agents: dict[str, AgentDefinition] | None = None
    setting_sources: list[SettingSource] | None = None


# SDK Control Protocol
class SDKControlInterruptRequest(TypedDict):
    subtype: Literal["interrupt"]


class SDKControlPermissionRequest(TypedDict):
    subtype: Literal["can_use_tool"]
    tool_name: str
    input: dict[str, Any]
    # TODO: Add PermissionUpdate type here
    permission_suggestions: list[Any] | None
    blocked_path: str | None


class SDKControlInitializeRequest(TypedDict):
    subtype: Literal["initialize"]
    hooks: dict[HookEvent, Any] | None


class SDKControlSetPermissionModeRequest(TypedDict):
    subtype: Literal["set_permission_mode"]
    # TODO: Add PermissionMode
    mode: str


class SDKHookCallbackRequest(TypedDict):
    subtype: Literal["hook_callback"]
    callback_id: str
    input: Any
    tool_use_id: str | None


class SDKControlMcpMessageRequest(TypedDict):
    subtype: Literal["mcp_message"]
    server_name: str
    message: Any


class SDKControlRequest(TypedDict):
    type: Literal["control_request"]
    request_id: str
    request: (
        SDKControlInterruptRequest
        | SDKControlPermissionRequest
        | SDKControlInitializeRequest
        | SDKControlSetPermissionModeRequest
        | SDKHookCallbackRequest
        | SDKControlMcpMessageRequest
    )


class ControlResponse(TypedDict):
    subtype: Literal["success"]
    request_id: str
    response: dict[str, Any] | None


class ControlErrorResponse(TypedDict):
    subtype: Literal["error"]
    request_id: str
    error: str


class SDKControlResponse(TypedDict):
    type: Literal["control_response"]
    response: ControlResponse | ControlErrorResponse

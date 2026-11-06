"""Short-term conversational memory for the Module 4 ROSA agent."""

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_ollama import ChatOllama


class ConversationMemory:
  """Keep a recent message window and summarize messages that leave it."""

  def __init__(self, llm: ChatOllama, window_messages: int = 6) -> None:
    if window_messages < 2:
      raise ValueError("window_messages must be at least 2")

    self.llm = llm
    self.window_messages = window_messages
    self.messages: list[BaseMessage] = []
    self.summary = ""

  async def add_turn(self, user_message: str, assistant_message: str) -> None:
    self.messages.extend(
      [HumanMessage(content=user_message), AIMessage(content=assistant_message)]
    )

    overflow = len(self.messages) - self.window_messages
    if overflow <= 0:
      return

    expired_messages = self.messages[:overflow]
    self.messages = self.messages[overflow:]
    transcript = self._format_messages(expired_messages)
    response = await self.llm.ainvoke(
      "Update the concise conversation summary using the older messages below. "
      "Preserve names, constraints, decisions, measurements, and unresolved tasks.\n\n"
      f"Existing summary:\n{self.summary or '(none)'}\n\n"
      f"Older messages:\n{transcript}\n\nUpdated summary:"
    )
    self.summary = str(response.content).strip()

  def render_context(self) -> str:
    recent = self._format_messages(self.messages) or "(none)"
    return (
      "Conversation summary:\n"
      f"{self.summary or '(none)'}\n\n"
      f"Recent message window (last {self.window_messages} messages):\n{recent}"
    )

  def status(self) -> str:
    return (
      f"Window: {len(self.messages)}/{self.window_messages} messages\n"
      f"Summary: {self.summary or '(empty)'}"
    )

  def clear(self) -> None:
    self.messages.clear()
    self.summary = ""

  @staticmethod
  def _format_messages(messages: list[BaseMessage]) -> str:
    lines = []
    for message in messages:
      role = "User" if isinstance(message, HumanMessage) else "Assistant"
      lines.append(f"{role}: {message.content}")
    return "\n".join(lines)
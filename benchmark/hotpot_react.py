"""
HotpotQA 任务上的 ReAct 智能体（非流式版本，便于批量评测）。
"""

import os
import re
import sys
import time
from typing import Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.llm import HelloAgentsLLM
from benchmark.prompt_hotpot import HOTPOT_REACT_PROMPT
from benchmark.wiki_tools import WikiEnv


class HotpotLLM(HelloAgentsLLM):
    """覆盖 think()：非流式 + 自定义 stop，避免一次吐出多个 step。
    内置最小调用间隔节流（默认 1.2s/次）和 429 指数退避重试。
    """

    def __init__(
        self,
        min_interval: float = 1.2,
        max_retries: int = 6,
        empty_retries: int = 3,
        empty_retry_wait: float = 3.0,
        **kw,
    ):
        super().__init__(**kw)
        self.min_interval = float(os.getenv("LLM_MIN_INTERVAL", min_interval))
        self.max_retries = max_retries
        self.empty_retries = empty_retries
        self.empty_retry_wait = empty_retry_wait
        self._last_call_ts: float = 0.0

    def _throttle(self):
        wait = self.min_interval - (time.time() - self._last_call_ts)
        if wait > 0:
            time.sleep(wait)

    def _call_once(self, prompt: str, temperature: float) -> Optional[str]:
        """单次调用 + 429 指数退避；不处理空返回。"""
        backoff = max(self.min_interval, 2.0)
        for attempt in range(self.max_retries + 1):
            self._throttle()
            self._last_call_ts = time.time()
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    stream=False,
                    stop=["\nObservation"],
                )
                return response.choices[0].message.content
            except Exception as e:
                msg = str(e)
                is_rate = (
                    "429" in msg
                    or "1302" in msg
                    or "rate" in msg.lower()
                    or "速率" in msg
                )
                if is_rate and attempt < self.max_retries:
                    print(
                        f"[LLM rate-limited] retry in {backoff:.1f}s (attempt {attempt + 1}/{self.max_retries})"
                    )
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 30.0)
                    continue
                print(f"[LLM error] {e}")
                return None
        return None

    def think_once(self, prompt: str, temperature: float = 0.0) -> Optional[str]:
        """带"空响应"重试：最多 empty_retries 次，每次等 empty_retry_wait 秒。
        每次重试都用稍微提高的温度，避免确定性贪心反复落到空输出。"""
        for attempt in range(self.empty_retries + 1):
            t = temperature + 0.2 * attempt
            resp = self._call_once(prompt, temperature=t)
            if resp and resp.strip():
                return resp
            if attempt < self.empty_retries:
                print(
                    f"[LLM empty] retry {attempt + 1}/{self.empty_retries} in {self.empty_retry_wait:.1f}s (temp={t + 0.2:.1f})"
                )
                time.sleep(self.empty_retry_wait)
        return None


class HotpotReActAgent:
    def __init__(self, llm: HotpotLLM, max_steps: int = 7, verbose: bool = False):
        self.llm = llm
        self.max_steps = max_steps
        self.verbose = verbose
        self.wiki = WikiEnv()

    def _build_prompt(self, question: str, history: str, next_step: int) -> str:
        # 在末尾显式给出下一步的 "Thought N:" 前缀，避免模型续写出 \nObservation 触发 stop。
        tail = f"\nThought {next_step}:"
        return (
            HOTPOT_REACT_PROMPT.replace("{question}", question).replace(
                "{history}", history
            )
            + tail
        )

    @staticmethod
    def _parse_step(text: str, step_i: int) -> Tuple[Optional[str], Optional[str]]:
        # 因为 prompt 末尾已经给了 "Thought N:"，模型直接续写 thought 内容，再换行写 "Action N:"。
        # 优先按这种"裸续写"格式解析；解析不到再回退到带 "Thought N:" 前缀的格式。
        bare_t = re.match(r"\s*(.*?)(?=\n\s*Action\b|\Z)", text, re.DOTALL)
        bare_a = re.search(
            rf"Action\s*{step_i}?\s*:\s*(.+?)(?:\n|$)", text, re.DOTALL
        )

        thought = bare_t.group(1).strip() if bare_t else None
        action = bare_a.group(1).strip() if bare_a else None

        # 回退：如果模型自己又写了 "Thought N:"（重复 prompt 末尾），重新提取
        t2 = re.search(
            rf"Thought\s*{step_i}\s*:\s*(.*?)(?=\n\s*Action\s*{step_i}\s*:|$)",
            text,
            re.DOTALL,
        )
        if t2:
            thought = t2.group(1).strip()
        if not action:
            a2 = re.search(r"Action[^:\n]*:\s*(.+?)(?:\n|$)", text, re.DOTALL)
            if a2:
                action = a2.group(1).strip()
        return thought, action

    @staticmethod
    def _parse_action(action: str) -> Tuple[Optional[str], Optional[str]]:
        m = re.match(r"(\w+)\s*\[(.*)\]\s*$", action, re.DOTALL)
        if m:
            return m.group(1).strip(), m.group(2).strip()
        return None, None

    def run(self, question: str) -> dict:
        self.wiki.reset()
        history = ""
        trace = []
        final_answer = None

        for step in range(1, self.max_steps + 1):
            prompt = self._build_prompt(question, history, step)
            response = self.llm.think_once(prompt)
            if not response:
                print(f"[step {step}] empty LLM response, abort")
                break

            thought, action = self._parse_step(response, step)
            if self.verbose:
                print(f"[step {step}] thought={thought!r} action={action!r}")

            if not action:
                print(f"[step {step}] no action parsed; raw={response!r}")
                trace.append({"step": step, "raw": response})
                break

            trace.append({"step": step, "thought": thought, "action": action})

            tool, arg = self._parse_action(action)
            if tool is None:
                observation = (
                    "Invalid action. Use Search[...], Lookup[...], or Finish[...]."
                )
            elif tool.lower() == "finish":
                final_answer = arg
                trace[-1]["observation"] = arg
                break
            elif tool.lower() == "search":
                observation = self.wiki.search(arg)
            elif tool.lower() == "lookup":
                observation = self.wiki.lookup(arg)
            else:
                observation = (
                    f"Invalid action {tool}. Use Search/Lookup/Finish."
                )

            trace[-1]["observation"] = observation
            if self.verbose:
                obs_show = observation if len(observation) < 200 else observation[:200] + "..."
                print(f"[step {step}] obs={obs_show}")

            history += (
                f"\nThought {step}: {thought or ''}"
                f"\nAction {step}: {action}"
                f"\nObservation {step}: {observation}"
            )

        return {
            "question": question,
            "answer": final_answer,
            "steps": len(trace),
            "trace": trace,
        }


if __name__ == "__main__":
    agent = HotpotReActAgent(HotpotLLM(), verbose=True)
    out = agent.run(
        "What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?"
    )
    print("FINAL:", out["answer"])

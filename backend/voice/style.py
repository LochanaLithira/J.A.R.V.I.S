"""How J.A.R.V.I.S speaks: calm, courteous, British, with a little dry understatement.

The tools return plain facts ("Opening notepad."). This file turns those facts into the line that
is spoken and shown ("Right away, sir. Opening Notepad."), with a few variations so it doesn't
sound robotic. It runs entirely on this laptop: no extra AI call, and nothing is sent back to the AI.

Privacy is unchanged: the line always starts from spoken_text(), so a private result (note text,
file names) is only ever replaced by a fixed "it's on your screen" line, never by its content.
"""
from __future__ import annotations

import random
import re
from datetime import datetime
from typing import Any, Callable

from .speak import PRIVATE_REPLY, spoken_text

_ADDRESS_RE = re.compile(r"[A-Za-z][A-Za-z .'-]{0,19}")

# (pattern for the plain fact, ways J.A.R.V.I.S may say it). {0}, {1}: parts of the fact; {a}: "sir".
_LINES: list[tuple[re.Pattern, tuple[str, ...]]] = [(re.compile(p), v) for p, v in [
    # ---- things done --------------------------------------------------------------------
    (r"Opening (.+)\.", ("Opening {0}, {a}.", "Right away, {a}. Opening {0}.", "Certainly, {a}. {0} is on its way.")),
    (r"Created note (.+)\.", ("Note {0} created, {a}.", "I've made a note called {0}, {a}.",
                              "Noted, {a}. It's saved as {0}.")),
    (r"Timer set for (.+)\.", ("Timer set for {0}, {a}.", "Very good, {a}. {0}, starting now.",
                               "Certainly, {a}. I'll let you know in {0}.")),
    (r"Reminder set for (.+)\.", ("Reminder set for {0}, {a}.", "Very good, {a}. I'll remind you in {0}.",
                                  "Consider it noted, {a}. I'll remind you in {0}.")),
    (r"Cancelled all (\d+) timers\.", ("All {0} timers cancelled, {a}.", "Done, {a}. I've cleared all {0} timers.")),
    (r"Cancelled (?:your|that) timer\.", ("Timer cancelled, {a}.", "Consider it cancelled, {a}.")),
    # ---- the safety gate ----------------------------------------------------------------
    (r"That action is not allowed\.", ("I'm afraid that isn't something I'm permitted to do, {a}.",
                                       "That falls outside what I'm allowed to do, {a}.")),
    (r"Cancelled\.", ("Very well, {a}. I've left it alone.", "Understood, {a}. Nothing has been changed.")),
    (r"I couldn't use those details\.", ("I'm afraid those details didn't quite work, {a}.",
                                         "Something in those details wasn't right, {a}. Perhaps try rephrasing.")),
    (r"Too many requests\. Please wait a moment\.", ("One moment, {a}. You're a step ahead of me.",
                                                     "If you'll give me a moment, {a}.")),
    (r"Something went wrong\.", ("My apologies, {a}. Something went wrong.",
                                 "I'm afraid something went awry, {a}.")),
    # ---- understanding ------------------------------------------------------------------
    (r"I didn't catch that\.", ("I'm sorry, {a}. I didn't quite catch that.", "Pardon me, {a}. Could you say that again?")),
    (r"That command is too long\.", ("That's rather a lot at once, {a}. Could you put it more briefly?",)),
    (r"The assistant gave an unusable answer\.", ("My apologies, {a}. I couldn't make sense of that one.",)),
    # ---- tools that couldn't help ----------------------------------------------------------
    (r"That app is not allowed\.", ("I'm afraid that application isn't on my approved list, {a}.",)),
    (r"I couldn't open that app\.", ("I'm afraid I couldn't open that application, {a}.",)),
    (r"A note with that title already exists\.", ("There's already a note by that name, {a}.",
                                                  "A note with that title already exists, {a}. Shall we pick another name?")),
    (r"That note name isn't allowed\.", ("I'm afraid I can't use that name for a note, {a}.",)),
    (r"You already have 10 timers\. Cancel one first\.", ("You already have ten timers running, {a}. "
                                                          "Shall I cancel one first?",)),
    (r"You have no timers to cancel\.", ("There are no timers running, {a}.",)),
    (r"I'm not sure which timer you mean\. Say its name, or say all\.",
     ("Which timer did you mean, {a}? You can name it, or say all.",)),
    (r"More than one timer matches\. Say its full name, or say all\.",
     ("More than one timer matches that, {a}. Its full name, or all?",)),
    (r"That folder isn't available\.", ("I'm afraid that folder isn't available, {a}.",)),
]]

_PRIVATE = ("It's on your screen, {a}.", "I've put it on your screen, {a}.", "Done, {a}. You'll find it on your screen.")
_TIMER = ("{A}, your timer has finished.", "Pardon the interruption, {a}. Your timer is up.", "Time's up, {a}.")
_REMINDER = ("{A}, a reminder for you. It's on your screen.", "Pardon the interruption, {a}. Your reminder is on your screen.")
_GREETING_END = ("All systems are online.", "At your service.", "Systems online and ready.")


class Style:
    def __init__(self, address: str = "sir", choose: Callable[[tuple[str, ...]], str] | None = None):
        self.address = address if _ADDRESS_RE.fullmatch(address or "") else "sir"
        self._choose = choose
        self._last: dict[int, str] = {}

    def _pick(self, options: tuple[str, ...]) -> str:
        """A variation, never the same one twice in a row for the same kind of line."""
        if self._choose is not None:
            pick = self._choose(options)
        else:
            fresh = [o for o in options if o != self._last.get(id(options))] or list(options)
            pick = random.choice(fresh)
        self._last[id(options)] = pick
        return pick

    def _fill(self, template: str, *parts: str) -> str:
        a = self.address
        return template.format(*parts, a=a, A=a[:1].upper() + a[1:])

    def dress(self, text: str) -> str:
        """Turn a plain, public fact into how J.A.R.V.I.S says it."""
        if not isinstance(text, str) or not text.strip():
            return text
        text = text.strip()
        for pattern, options in _LINES:
            m = pattern.fullmatch(text)
            if m:
                parts = [p[:1].upper() + p[1:] if i == 0 and p.islower() and " " not in p else p
                         for i, p in enumerate(m.groups())]
                return self._fill(self._pick(options), *parts)
        # Anything else that is one plain sentence gets a courteous address, unless it has one.
        if (text.endswith(".") and text.count(".") == 1 and self.address.lower() not in text.lower()
                and len(text) <= 120):
            return text[:-1] + f", {self.address}."
        return text

    def line(self, result: Any) -> str:
        """What J.A.R.V.I.S says for a gate result. FAIL CLOSED: anything not explicitly public
        becomes a fixed "it's on your screen" line, exactly as spoken_text() decides."""
        base = spoken_text(result)
        if base == PRIVATE_REPLY and getattr(result, "private", True) is not False:
            return self._fill(self._pick(_PRIVATE))
        return self.dress(base)

    def timer_line(self, has_label: bool) -> str:
        """The label is the user's own words, so it is never part of the spoken line."""
        return self._fill(self._pick(_REMINDER if has_label else _TIMER))

    def greeting(self, now: datetime | None = None) -> str:
        hour = (now or datetime.now()).hour
        if 5 <= hour < 12:
            opening = "Good morning, {a}."
        elif 12 <= hour < 17:
            opening = "Good afternoon, {a}."
        elif 17 <= hour < 22:
            opening = "Good evening, {a}."
        else:
            opening = "Working late, {a}?"
        return self._fill(opening) + " " + self._pick(_GREETING_END)

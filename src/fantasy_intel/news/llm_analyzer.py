"""LLM context extractor and depth-chart impact analyzer for breaking news."""

from __future__ import annotations
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional
import requests

from fantasy_intel.analytics.depth_chart import DepthChartManager

logger = logging.getLogger(__name__)

# Heuristic patterns for robust fallback parsing
SEVERITY_PATTERNS = {
    "IR": [r"\bplaced on ir\b", r"\binjured reserve\b", r"\bout for season\b", r"\btorn acl\b", r"\btorn achilles\b"],
    "Out": [r"\bruled out\b", r"\bwill not play\b", r"\bmiss \d+ weeks?\b", r"\bsidelined\b", r"\bout vs\b", r"\bout against\b"],
    "Doubtful": [r"\bdoubtful\b", r"\bunlikely to play\b"],
    "Questionable": [r"\bquestionable\b", r"\bgame-time decision\b", r"\blimited in practice\b", r"\bdid not practice\b"],
}

TIMEFRAME_PATTERNS = [
    (r"\bout for season\b", "Season-ending"),
    (r"\b(\d+[\s-]to[\s-]\d+\s+weeks?)\b", r"\1"),
    (r"\b(\d+\s+weeks?)\b", r"\1"),
    (r"\bseveral weeks\b", "Multi-week absence (3-6 weeks)"),
    (r"\bday-to-day\b", "Day-to-day"),
    (r"\bweek-to-week\b", "Week-to-week (1-3 weeks)"),
    (r"\bgame-time decision\b", "Game-time decision"),
]

BODY_PARTS = [
    "hamstring", "concussion", "ankle", "knee", "calf", "shoulder",
    "quad", "achilles", "groin", "foot", "wrist", "ribs", "thumb", "elbow"
]


class LLMNewsAnalyzer:
    """Extracts structured fantasy intelligence, injury severity, and depth chart promotions from news."""

    def __init__(self, depth_manager: Optional[DepthChartManager] = None):
        self.depth_manager = depth_manager or DepthChartManager()
        self.gemini_key = os.environ.get("GEMINI_API_KEY")
        self.openai_key = os.environ.get("OPENAI_API_KEY")

    def analyze_news_items(self, articles: List[Dict[str, Any]], limit: int = 10) -> List[Dict[str, Any]]:
        """Analyzes a list of news articles or injury items into structured fantasy impact intelligence."""
        results = []
        for article in articles[:limit]:
            analyzed = self.analyze_single(article)
            if analyzed:
                results.append(analyzed)
        return results

    def analyze_single(self, article: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyzes a single news article using LLM (if configured) or intelligent heuristic fallback."""
        headline = article.get("title", "")
        summary = article.get("summary", "")
        text = f"{headline} {summary}".strip()
        if not text:
            return None

        # Attempt LLM analysis if API key is present
        if self.gemini_key:
            llm_result = self._call_gemini_api(headline, summary)
            if llm_result:
                return self._enrich_analysis(llm_result, article)
        elif self.openai_key:
            llm_result = self._call_openai_api(headline, summary)
            if llm_result:
                return self._enrich_analysis(llm_result, article)

        # Fallback to intelligent deterministic parser
        return self._heuristic_analysis(article)

    def _call_gemini_api(self, headline: str, summary: str) -> Optional[Dict[str, Any]]:
        """Call Gemini API via REST for structured fantasy injury extraction."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={self.gemini_key}"
        prompt = (
            "You are an expert NFL fantasy football intelligence analyst. "
            "Analyze the following NFL news headline and summary. "
            "Extract structured data in JSON format with these exact keys: "
            "is_injury_or_role_change (bool), player_name (str or null), team (str or null), position (str or null), "
            "injury_details (str or null), status (Out/IR/Doubtful/Questionable/Active/Unknown), "
            "expected_missed_time (str or null), promoted_beneficiary_name (str or null), "
            "opportunity_impact (str), waiver_urgency (CRITICAL/HIGH/MODERATE/LOW), fantasy_action (str).\n\n"
            f"Headline: {headline}\n"
            f"Summary: {summary}\n\n"
            "Respond ONLY with valid JSON."
        )

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json"},
        }
        try:
            resp = requests.post(url, json=payload, timeout=8)
            if resp.status_code == 200:
                data = resp.json()
                text_out = data["candidates"][0]["content"]["parts"][0]["text"]
                return json.loads(text_out)
        except Exception as e:
            logger.warning(f"Gemini API analysis failed: {e}")
        return None

    def _call_openai_api(self, headline: str, summary: str) -> Optional[Dict[str, Any]]:
        """Call OpenAI API via REST for structured fantasy injury extraction."""
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.openai_key}", "Content-Type": "application/json"}
        prompt = (
            "You are an expert NFL fantasy football analyst. "
            "Extract structured data in JSON with keys: "
            "is_injury_or_role_change (bool), player_name, team, position, injury_details, status, "
            "expected_missed_time, promoted_beneficiary_name, opportunity_impact, waiver_urgency, fantasy_action.\n\n"
            f"Headline: {headline}\n"
            f"Summary: {summary}\n\n"
            "Return valid JSON only."
        )
        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=8)
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                return json.loads(content)
        except Exception as e:
            logger.warning(f"OpenAI API analysis failed: {e}")
        return None

    def _heuristic_analysis(self, article: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Intelligent deterministic parsing connecting news text to depth charts."""
        text = f"{article.get('title', '')} {article.get('summary', '')}"
        lower_text = text.lower()

        # 1. Resolve Player Name from Depth Chart / DB
        player_info = None
        if article.get("player_name"):
            player_info = self.depth_manager.find_player(article["player_name"])
        else:
            self.depth_manager._ensure_depth_chart()
            for p in self.depth_manager._players.values():
                fname = p.get("full_name")
                if fname and len(fname) > 5 and fname.lower() in lower_text:
                    player_info = self.depth_manager.find_player(fname)
                    if player_info:
                        break

        # 2. Detect Injury Signals
        detected_injury = None
        for bp in BODY_PARTS:
            if bp in lower_text:
                detected_injury = f"{bp.capitalize()} injury"
                break

        # 3. Detect Status
        detected_status = article.get("injury_status") or "Active"
        if detected_status == "Active":
            for status_val, patterns in [
                ("IR", [r"\bplaced on ir\b", r"\binjured reserve\b", r"\bout for season\b", r"\btorn acl\b", r"\btorn achilles\b"]),
                ("Out", [r"\bruled out\b", r"\bwill not play\b", r"\bout vs\b", r"\bout against\b", r"\b is out\b", r"\b out,\b", r"\bout with\b", r"\bmiss \d+ weeks?\b", r"\bsidelined\b"]),
                ("Doubtful", [r"\bdoubtful\b", r"\bunlikely to play\b"]),
                ("Questionable", [r"\bquestionable\b", r"\bgame-time decision\b", r"\blimited in practice\b", r"\bdid not practice\b"]),
            ]:
                for pat in patterns:
                    if re.search(pat, lower_text):
                        detected_status = status_val
                        break
                if detected_status != "Active":
                    break

        # If no player found, or no injury/role signal, skip this non-injury article
        if not player_info and not detected_injury and detected_status == "Active":
            return None

        # 4. Detect Missed Time
        missed_time = "Day-to-day"
        if detected_status in ["IR", "Out"]:
            missed_time = "1-3 weeks (Out)"
        for pat, desc in TIMEFRAME_PATTERNS:
            match = re.search(pat, lower_text)
            if match:
                missed_time = desc if "\\" not in desc else match.expand(desc)
                break

        # 5. Resolve Next Man Up
        beneficiary = None
        player_name = player_info["name"] if player_info else article.get("player_name", "Unknown Player")
        team = player_info["team"] if player_info else article.get("team", "N/A")
        pos = player_info["position"] if player_info else article.get("position", "N/A")
        depth_order = player_info["depth_order"] if player_info else 99

        if player_info:
            beneficiary = self.depth_manager.get_next_man_up(player_info["name"])

        # 6. Determine Waiver Urgency & Actionable Advice
        is_starter = depth_order == 1 or (pos == "WR" and depth_order <= 2)
        if detected_status in ["Out", "IR"] and is_starter:
            urgency = "CRITICAL (Top Waiver Priority)"
            action = f"Add {beneficiary['name']} immediately ({team} {pos}); starter {player_name} is sidelined." if beneficiary else f"Starter {player_name} is ruled out; bench immediately."
            impact = f"Vacates high-leverage starter touches on {team}. {beneficiary['name'] if beneficiary else 'Backup'} projects for major volume expansion."
        elif detected_status in ["Doubtful", "Questionable"] and is_starter:
            urgency = "HIGH (Strong Stash / Prepare Sub)"
            action = f"Stash {beneficiary['name']} as protective insurance in case {player_name} is inactive." if beneficiary else f"Monitor {player_name} game-time status."
            impact = f"Potential starter absence creates high-ceiling opportunity for {beneficiary['name'] if beneficiary else 'backups'}."
        else:
            urgency = "MODERATE (Roster Watch)"
            action = f"Keep an eye on depth chart rotation for {team}."
            impact = f"Minor depth chart adjustment on {team}."

        return {
            "title": article.get("title", ""),
            "source": article.get("source", "NFL Wire"),
            "player_name": player_name,
            "team": team,
            "position": pos,
            "depth_order": depth_order,
            "injury_details": detected_injury or article.get("injury_body_part") or "Injury",
            "status": detected_status,
            "expected_missed_time": missed_time,
            "promoted_beneficiary": beneficiary.get("name") if beneficiary else None,
            "beneficiary_details": beneficiary,
            "opportunity_impact": impact,
            "waiver_urgency": urgency,
            "fantasy_action": action,
            "summary": article.get("summary", ""),
        }

    def _enrich_analysis(self, llm_data: Dict[str, Any], article: Dict[str, Any]) -> Dict[str, Any]:
        """Cross-references LLM extraction with depth chart database."""
        pname = llm_data.get("player_name") or ""
        player_info = self.depth_manager.find_player(pname) if pname else None
        beneficiary = None

        if player_info:
            beneficiary = self.depth_manager.get_next_man_up(player_info["name"])

        return {
            "title": article.get("title", ""),
            "source": article.get("source", "NFL Wire"),
            "player_name": pname or "Unknown Player",
            "team": llm_data.get("team") or (player_info["team"] if player_info else "N/A"),
            "position": llm_data.get("position") or (player_info["position"] if player_info else "N/A"),
            "depth_order": player_info["depth_order"] if player_info else 99,
            "injury_details": llm_data.get("injury_details") or "Injury",
            "status": llm_data.get("status") or "Active",
            "expected_missed_time": llm_data.get("expected_missed_time") or "Day-to-day",
            "promoted_beneficiary": (beneficiary.get("name") if beneficiary else None) or llm_data.get("promoted_beneficiary_name"),
            "beneficiary_details": beneficiary,
            "opportunity_impact": llm_data.get("opportunity_impact") or "Opportunity shift across depth chart.",
            "waiver_urgency": llm_data.get("waiver_urgency") or "MODERATE",
            "fantasy_action": llm_data.get("fantasy_action") or "Monitor team practice reports.",
            "summary": article.get("summary", ""),
        }

from crewai import Agent

class EscalationAgent:
    def __init__(self, log_file="escalation_log.txt"):
        self.log_file = log_file
        self.agent = Agent(
            name="EscalationAgent",
            role="Handles low confidence or 'Other' cases",
            goal="Escalate email if required",
            backstory="You are responsible for identifying and escalating emails that cannot be confidently categorized or require special attention."
        )

    def run(self, input_data):
        confidence = input_data["confidence"]
        category = input_data["predicted_category"]
        if category == "Other" or confidence < 0.6:
            reason = "Low confidence or unknown category"
            with open(self.log_file, "a") as f:
                f.write(f"{input_data['email_text']} -> {reason}\n")
            return {
                "status": "escalated",
                "reason": reason,
                "logged_to": self.log_file
            }
        return {"status": "not escalated"}

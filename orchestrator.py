from agents.email_classifier import EmailClassifierAgent
from agents.response_generator import ResponseGeneratorAgent
from agents.escalation_agent import EscalationAgent

class SmartEmailCrew:
    def __init__(self):
        self.classifier = EmailClassifierAgent()
        self.responder = ResponseGeneratorAgent()
        self.escalator = EscalationAgent()

    def process(self, email_text):
        classification_result = self.classifier.run(email_text)

        if classification_result["confidence"] >= 0.6 and classification_result["predicted_category"] != "Other":
            response = self.responder.run(classification_result)
            return {
                "agent": "Response Generator",
                "input": classification_result,
                "output": response
            }
        else:
            escalation = self.escalator.run(classification_result)
            return {
                "agent": "Escalation Agent",
                "input": classification_result,
                "output": escalation
            }

# Sample usage
if __name__ == "__main__":
    crew = SmartEmailCrew()
    test = crew.process("I need to update my salary account details.")
    print(test)

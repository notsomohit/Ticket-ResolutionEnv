# Support Ticket Environment

## Problem Description
This environment simulates a Customer Support Ticket Resolution process with real-world constraints, SLA deadlines, and resource management. Agents must handle tickets for Free, Premium, and Enterprise customers while optimizing for speed, accuracy, and senior agent availability.

## Motivation
Real support systems aren't just about correctness; they are about **efficiency** and **prioritization**. This environment introduces:
- **SLA Deadlines:** Tickets have step limits. Missing them incurs heavy penalties.
- **Resource Constraints:** Only a limited number of "Tier 2" assignments are allowed per session.
- **Sentiment & Priority:** Angry enterprise customers demand faster action, but simple issues shouldn't be over-escalated.

## Action & Observation Space
**Observation:**
- `priority`: (1-3) 1: Low, 2: Medium, 3: High.
- `sla_remaining`: Steps left before SLA breach.
- `sentiment`: "angry", "neutral", "happy".
- `queue_length`: Number of tickets remaining in the queue.

**Action:**
- `action_type`: "set_category", "set_assignee", "set_resolution", "close_ticket".
- `value`: Appropriate string values for each field.

## Tasks & Examples
### 1. Easy
- **Input:** Billing refund request.
- **Good Action:** Set category to `billing`, assignee to `billing_team`, resolution to `issue_refund`, and close.
- **Outcome:** High reward, 100% accuracy.

### 2. Medium
- **Input:** Angry premium customer with a 500 error.
- **Good Action:** Fast escalation to `tier2_support`.
- **Outcome:** SLA bonus for quick resolution.

### 3. Hard (Ambiguity & Constraints)
- **Input:** Enterprise user screaming about being locked out.
- **Challenge:** The tight SLA (5 steps) tempts escalation, but "Tier 2" is limited. The correct path is a standard `account` resolution by `tier1_support`.
- **Outcome:** Test of reasoning vs reactive behavior.

## Rewards (V2)
- **Efficiency:** -0.05 per step.
- **Accuracy:** +1.0 per correct field on close; -0.5 per wrong field.
- **SLA Bonus:** +0.5 for meeting deadlines; -1.0 for breaches.
- **Constraint Penalty:** -2.0 for trying to use Tier 2 when slots are empty.
- **Anti-Loop:** Geometric penalty for repeated actions.

## Grader (V2)
Produces a score [0.0 - 1.0] based on:
- **Accuracy (40%):** Are fields correct?
- **SLA (30%):** Did it meet the time constraints?
- **Efficiency (30%):** Did it take the optimal path (4 steps)?

## Setup & Usage
```bash
pip install -r requirements.txt
python inference.py --task hard --model gpt-4o
```

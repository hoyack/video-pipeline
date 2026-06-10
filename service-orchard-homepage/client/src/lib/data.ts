// Service Orchard Homepage Data
// All content sourced from the homepage spec

export const roleCards = [
  {
    id: "hvac-dispatcher",
    title: "HVAC Dispatcher",
    industry: "HVAC",
    pain: "Calls, schedule changes, tech updates, angry customers, and follow-ups all hit at once.",
    cta: "See HVAC Dispatcher Skills",
  },
  {
    id: "plumbing-admin",
    title: "Plumbing Office Admin",
    industry: "Plumbing",
    pain: "Emergency calls, quote follow-ups, customer updates, and schedule changes can pile up fast.",
    cta: "See Plumbing Admin Skills",
  },
  {
    id: "electrical-estimator",
    title: "Electrical Estimator",
    industry: "Electrical",
    pain: "Turn messy notes, scope details, and vendor quotes into cleaner estimate drafts.",
    cta: "See Electrical Estimator Skills",
  },
  {
    id: "construction-estimator",
    title: "Construction Estimator",
    industry: "Construction",
    pain: "Clean up scopes, compare quotes, chase missing details, and draft bid follow-ups faster.",
    cta: "See Construction Estimator Skills",
  },
  {
    id: "warranty-coordinator",
    title: "Homebuilder Warranty Coordinator",
    industry: "Homebuilding",
    pain: "Track warranty requests, summarize issues, follow up with trades, and keep buyers updated.",
    cta: "See Warranty Coordinator Skills",
  },
  {
    id: "transaction-coordinator",
    title: "Real Estate Transaction Coordinator",
    industry: "Real Estate",
    pain: "Track deadlines, chase documents, summarize reports, and keep every party moving.",
    cta: "See Transaction Coordinator Skills",
  },
];

export const whatWeDoPoints = [
  {
    title: "Drafts",
    description: "Turn repeated emails, customer updates, and follow-ups into ready-to-review drafts.",
    icon: "FileText",
  },
  {
    title: "Summaries",
    description: "Summarize notes, reports, calls, emails, job updates, and project details.",
    icon: "ListChecks",
  },
  {
    title: "Queues",
    description: "Keep follow-ups, missing items, and next steps from slipping through the cracks.",
    icon: "Layers",
  },
  {
    title: "Reports",
    description: "Generate daily, weekly, or job-specific summaries without rebuilding them from scratch.",
    icon: "BarChart3",
  },
  {
    title: "Checklists",
    description: "Turn messy work into repeatable steps, reminders, and status updates.",
    icon: "CheckSquare",
  },
];

export const automatableCategories = [
  {
    title: "Repeated Messages",
    description: "Emails, texts, customer updates, vendor follow-ups, appointment confirmations, and reminders.",
    icon: "Mail",
  },
  {
    title: "Notes and Summaries",
    description: "Technician notes, inspection reports, project updates, call notes, meeting notes, and field updates.",
    icon: "StickyNote",
  },
  {
    title: "Scheduling Support",
    description: "Appointment confirmations, availability checks, schedule summaries, reminder drafts, and daily briefs.",
    icon: "Calendar",
  },
  {
    title: "Follow-Up Management",
    description: "Open quotes, missing documents, unscheduled estimates, customer no-responses, and vendor reminders.",
    icon: "Bell",
  },
  {
    title: "Document and Form Help",
    description: "Intake forms, reports, checklists, proposals, scope summaries, and handoff notes.",
    icon: "FileStack",
  },
  {
    title: "CRM and Admin Cleanup",
    description: "Contact updates, task summaries, status notes, pipeline updates, and record cleanup.",
    icon: "Database",
  },
];

export const skillPacks = [
  {
    id: "hvac-dispatcher",
    title: "HVAC Dispatcher Skill Pack",
    description: "Built for the daily chaos of dispatch boards, customer calls, technician updates, and schedule changes.",
    skills: [
      "Missed call follow-up drafts",
      "Appointment confirmation messages",
      "Technician note cleanup",
      "Daily dispatch summary",
      "Customer ETA update drafts",
      "Maintenance plan reminders",
      "Review request messages",
    ],
    cta: "Start HVAC Dispatcher Scan",
  },
  {
    id: "construction-estimator",
    title: "Construction Estimator Skill Pack",
    description: "Built for turning notes, scopes, vendor quotes, and missing information into cleaner estimate workflows.",
    skills: [
      "Estimate draft generation",
      "Scope summary cleanup",
      "Vendor quote comparison",
      "Missing information request drafts",
      "Bid follow-up messages",
      "Proposal language drafts",
      "Change order summary drafts",
    ],
    cta: "Start Estimator Scan",
  },
  {
    id: "transaction-coordinator",
    title: "Real Estate Transaction Coordinator Skill Pack",
    description: "Built for deadline tracking, document chasing, inspection summaries, and client communication.",
    skills: [
      "Deadline checklist tracking",
      "Inspection report summaries",
      "Missing document reminders",
      "Client update drafts",
      "Title/lender/agent coordination messages",
      "Transaction status summaries",
      "Closing timeline briefs",
    ],
    cta: "Start Transaction Coordinator Scan",
  },
  {
    id: "warranty-coordinator",
    title: "Homebuilder Warranty Coordinator Skill Pack",
    description: "Built for warranty requests, buyer updates, vendor coordination, and open issue tracking.",
    skills: [
      "Warranty request intake summaries",
      "Buyer update drafts",
      "Trade/vendor follow-up messages",
      "Issue categorization",
      "Appointment coordination",
      "Open warranty item reports",
      "Escalation summaries",
    ],
    cta: "Start Warranty Coordinator Scan",
  },
];

export const howItWorksSteps = [
  {
    step: 1,
    title: "Pick your job",
    description: "Start with your actual role, not a generic automation category.",
  },
  {
    step: 2,
    title: "Tell us what slows you down",
    description: "Share the repetitive tasks, tools, messages, reports, and follow-ups that eat up your day.",
  },
  {
    step: 3,
    title: "Get a Job Automation Scan",
    description: "We identify the first practical AI skills that could help.",
  },
  {
    step: 4,
    title: "Build your first Skill Pack",
    description: "Start with a small bundle of useful skills around your role.",
  },
  {
    step: 5,
    title: "Grow from there",
    description: "Add more skills, upgrade the role, or escalate to Thunderstaff if the workflow should be managed for the business.",
  },
];

export const offerLadder = [
  {
    step: 1,
    title: "Free Job Automation Scan",
    description: "Find the repetitive tasks inside your role and identify the best first AI skills to build.",
    cta: "Start Free Scan",
  },
  {
    step: 2,
    title: "Starter Skill Pack",
    description: "Install a small bundle of AI skills for one role, such as follow-up drafts, summaries, checklists, or reporting helpers.",
    cta: "Ask About Skill Packs",
  },
  {
    step: 3,
    title: "Role Upgrade",
    description: "Redesign a role around AI-assisted workflows, SOPs, templates, reporting, and repeatable output.",
    cta: "Upgrade a Role",
  },
  {
    step: 4,
    title: "Thunderstaff Escalation",
    description: "When the business wants the workflow managed end-to-end, Thunderstaff can automate, monitor, and operate the process.",
    cta: "Explore Thunderstaff",
  },
  {
    step: 5,
    title: "Hoyack Custom Build",
    description: "When the business needs custom software, integrations, dashboards, or a deeper AI system, Hoyack can build the infrastructure.",
    cta: "Talk to Hoyack",
  },
];

export const navLinks = [
  { label: "Jobs", href: "#roles" },
  { label: "Skill Packs", href: "#skill-packs" },
  { label: "How It Works", href: "#how-it-works" },
  { label: "Thunderstaff", href: "#offer-ladder" },
];

export const footerColumns = [
  {
    title: "Service Orchard",
    links: [
      { label: "Home", href: "#" },
      { label: "Jobs", href: "#roles" },
      { label: "Skill Packs", href: "#skill-packs" },
      { label: "Job Automation Scan", href: "#scan" },
      { label: "How It Works", href: "#how-it-works" },
    ],
  },
  {
    title: "Job Roles",
    links: [
      { label: "HVAC Dispatcher", href: "#roles" },
      { label: "Plumbing Office Admin", href: "#roles" },
      { label: "Electrical Estimator", href: "#roles" },
      { label: "Construction Estimator", href: "#roles" },
      { label: "Homebuilder Warranty Coordinator", href: "#roles" },
      { label: "Real Estate Transaction Coordinator", href: "#roles" },
    ],
  },
  {
    title: "Company",
    links: [
      { label: "Thunderstaff", href: "#offer-ladder" },
      { label: "Hoyack", href: "#offer-ladder" },
      { label: "Contact", href: "#scan" },
    ],
  },
  {
    title: "Legal",
    links: [
      { label: "Privacy Policy", href: "#" },
      { label: "Terms", href: "#" },
    ],
  },
];

"""Field catalogue for structured extraction (Section 8 of the assignment).

Each field carries: the group (extraction specialist) that owns it, retrieval queries, preferred document
types, whether addendums may override it, and a format hint used by the validator.
Adding a field = adding one entry here; no other code changes are needed.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FieldSpec:
    name: str
    description: str
    group: str
    queries: list[str]
    prefer_doc_types: list[str] = field(default_factory=list)
    addendum_sensitive: bool = False
    format: str = "text"  # text | datetime | bool_text | list | contact | summary
    required: bool = True


GROUPS = {
    "dates_logistics": "Dates & Logistics",
    "commercial_legal": "Commercial & Legal",
    "product_specs": "Product & Specs",
    "summary": "Summary",
}

FIELDS: list[FieldSpec] = [
    FieldSpec("Bid Number", "Official solicitation / RFP identifier (e.g. JA-207652 or a PORFP number).", "dates_logistics",
              ["solicitation number RFP number bid number", "PORFP number project number reference number"],
              ["bid_page", "rfp"]),
    FieldSpec("Title", "Full title of the bid or RFP.", "dates_logistics",
              ["title of the solicitation RFP", "request for proposal title"], ["bid_page", "rfp"]),
    FieldSpec("Due Date", "Final submission deadline including time and time zone. Must reflect addendum changes.", "dates_logistics",
              ["proposal due date and time", "submission deadline closing date", "new due date extended"],
              ["addendum", "rfp", "bid_page"], addendum_sensitive=True, format="datetime"),
    FieldSpec("Bid Submission Type", "How the bid is submitted (electronic portal, sealed paper, email, etc.).", "dates_logistics",
              ["how to submit proposal electronic portal sealed envelope", "bid submission instructions online system email fax"],
              ["rfp", "bid_page"], addendum_sensitive=True),
    FieldSpec("Term of Bid", "Contract duration and any renewal options.", "commercial_legal",
              ["term of contract initial term renewal options", "length of contract years extension"], ["rfp"], addendum_sensitive=True),
    FieldSpec("Pre Bid Meeting", "Date, time, location and whether attendance is mandatory, or 'None'.", "dates_logistics",
              ["pre-proposal meeting pre-bid conference date time location", "prebid conference mandatory attendance"],
              ["rfp", "bid_page"], addendum_sensitive=True, format="datetime_or_none"),
    FieldSpec("Installation", "Whether installation / deployment / imaging services are required.", "product_specs",
              ["installation deployment services imaging required", "white glove deployment setup asset tagging etching autopilot enrollment"],
              ["rfp", "addendum"], addendum_sensitive=True, format="bool_text"),
    FieldSpec("Bid Bond Requirement", "Required bid bond or security, amount or percentage, or 'Not required'.", "commercial_legal",
              ["bid bond requirement amount percentage", "bid security surety performance bond"], ["rfp"], addendum_sensitive=True),
    FieldSpec("Delivery Date", "Required delivery date or delivery window after award.", "dates_logistics",
              ["delivery date delivery within days of award", "when must the equipment be delivered lead time ship date"],
              ["rfp", "addendum"], addendum_sensitive=True),
    FieldSpec("Payment Terms", "Payment terms (e.g. Net 30) and/or invoicing conditions: where invoices are sent, when they are due after delivery, what they must include. If only invoicing instructions exist, report those.", "commercial_legal",
              ["payment terms net 30 invoicing", "invoicing instructions email invoices within days of delivery accounts payable", "invoice shall include purchase order number"],
              ["rfp"], addendum_sensitive=True),
    FieldSpec("Any Additional Documentation Required", "Affidavits, forms, certificates, insurance, W-9, etc. that must be submitted.", "commercial_legal",
              ["required forms affidavits certificates attachments to submit with proposal", "W-9 form 1295 insurance certificate mercury affidavit contract affidavit required documents"],
              ["rfp", "affidavit", "addendum"], addendum_sensitive=True, format="list"),
    FieldSpec("MFG for Registration", "Manufacturer(s) the bidder must be registered / authorized with.", "product_specs",
              ["manufacturer authorized reseller letter of authorization", "OEM manufacturer name factory authorized"], ["rfp", "specs"]),
    FieldSpec("Contract or Cooperative to use", "Referenced cooperative / state contract vehicle, if any.", "commercial_legal",
              ["master contract cooperative purchasing contract vehicle", "piggyback interlocal cooperative state contract number"],
              ["rfp", "bid_page"]),
    FieldSpec("Model_no", "Model number(s) of the requested product(s).", "product_specs",
              ["model number of requested product", "model # SI# laptop dock monitor model"], ["rfp", "specs", "bid_page"], format="list"),
    FieldSpec("Part_no", "Part / SKU number(s) of the requested product(s).", "product_specs",
              ["part number SKU of requested products", "SKU description part no"], ["specs", "rfp"], format="list"),
    FieldSpec("Product", "Product name(s) and quantity.", "product_specs",
              ["product name and quantity requested", "line items qty each laptops desktops tablets monitors"],
              ["rfp", "bid_page", "specs"], addendum_sensitive=True, format="list"),
    FieldSpec("contact_info", "Procurement contact for the solicitation (the buyer / procurement officer / agency POC named for questions and submissions), with name, email and phone. Prefer the buyer or procurement services contact over other departments.", "dates_logistics",
              ["Buyer Email header information", "Contact Information procurement services phone email",
               "agency point of contact procurement officer email telephone", "questions concerning this solicitation should be addressed to"],
              ["bid_page", "rfp"], format="contact"),
    FieldSpec("company_name", "Issuing organization / agency name.", "dates_logistics",
              ["issuing organization agency name", "purchasing department district office name"], ["bid_page", "rfp"]),
    FieldSpec("Product Specification", "Key technical specifications (CPU, RAM, storage, display, warranty, etc.).", "product_specs",
              ["technical specifications processor memory storage display", "minimum requirements CPU RAM SSD screen warranty years"],
              ["specs", "rfp", "addendum"], addendum_sensitive=True, format="text"),
    FieldSpec("Bid Summary", "A concise (3-6 sentence) summary of the bid, generated by the system.", "summary",
              ["purpose of this request for proposal scope overview", "description of the solicitation"], ["rfp", "bid_page"],
              format="summary"),
]

FIELD_BY_NAME = {f.name: f for f in FIELDS}


def fields_in_group(group: str) -> list[FieldSpec]:
    return [f for f in FIELDS if f.group == group]
"""Structured output schema for the single agent.

One agent researches a company's physical sites and returns them directly -- no separate
verification pass (see agent.py's docstring for why that step was removed)."""

from pydantic import BaseModel, Field


class CandidateAddress(BaseModel):
    organization_name: str | None = None
    legal_entity: str | None = Field(
        default=None,
        description="The specific operating parent/subsidiary that owns or runs this site, if "
        "different from organization_name -- state the ownership relationship (wholly owned, "
        "majority/minority stake, joint venture) in ownership_type rather than here.",
    )
    site_name: str | None = Field(default=None, description="Official name of the facility, if published.")
    site_type: str | None = Field(
        default=None,
        description="Comma-separated list of every function this site serves, e.g. 'Manufacturing "
        "Plant, R&D, Office' -- list all of them rather than picking just one when a site is "
        "multi-purpose.",
    )
    street_address: str | None = Field(
        default=None, description="Building number, street, and industrial park/estate if any."
    )
    city: str | None = None
    state_province: str | None = None
    postal_code: str | None = None
    country: str | None = None
    ownership_type: str | None = Field(
        default=None, description="Wholly owned, leased, subsidiary, joint venture (name the partner "
        "and ownership % if available), or similar."
    )
    operational_status: str | None = Field(
        default=None, description="Active, under construction, planned, idle, or closed -- leave blank "
        "if not confirmable, never guess."
    )
    status_as_of: str | None = Field(default=None, description="Date the operational_status was last confirmed by a source.")
    source_url: str = Field(description="URL of the page/document that supports this address.")
    evidence_quote: str = Field(description="Short verbatim snippet from the source supporting this address.")

    def full_address(self) -> str:
        """Every address component joined into one string, for token-based matching/dedupe --
        the individual fields stay separate everywhere else (CSV columns, comparisons)."""
        parts = [self.street_address, self.city, self.state_province, self.postal_code, self.country]
        return ", ".join(p for p in parts if p)


class SiteExtractionResult(BaseModel):
    candidates: list[CandidateAddress] = Field(default_factory=list)
    company_domains: list[str] = Field(
        default_factory=list,
        description="Every domain you confirmed is operated BY the company, its subsidiaries or "
        "its acquired brands -- the main site first, then country/regional sites and brand sites "
        "(e.g. ['acme.com', 'acme.co.uk', 'acme.com.my', 'acmebrand.com']). Scoring treats a "
        "source on any of these as first-party evidence, which is how a country contact page gets "
        "the credit it deserves instead of being scored like a scraped directory. Include only "
        "domains you actually saw serving the company's own content; never a directory, "
        "aggregator, registry or news site.",
    )
    notes: str = Field(
        default="",
        description="Free-text caveats. State the company's primary official domain here (e.g. "
        "'3m.com') as well, since scoring falls back to it when company_domains is empty.",
    )

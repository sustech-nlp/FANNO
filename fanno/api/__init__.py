"""Azure OpenAI API client utilities."""

from fanno.api.client import (
    AzureAPIClient,
    get_client,
    get_endpoints,
    select_endpoint,
)

__all__ = ["AzureAPIClient", "get_client", "get_endpoints", "select_endpoint"]

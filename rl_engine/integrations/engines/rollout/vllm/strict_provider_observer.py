# SPDX-License-Identifier: Apache-2.0
"""vLLM GDN candidate collection; no implicit qualification of opaque dispatch."""

from rl_engine.contracts.strict_provider import StrictProviderError
from rl_engine.integrations.common.gdn_provider_observation import (
    observe_gdn_instance,
    observe_gdn_metadata,
    observe_gdn_states,
)


class VllmGDNObserver:
    """Inspect a bound instance and the actual state views supplied by its hook.

    This collector intentionally cannot produce an executable strict call:
    the version-specific native/JIT dispatch adapter has not been qualified.
    """

    def observe_bound_provider(self, provider):
        return observe_gdn_instance(provider)

    def observe_call(self, provider, metadata, *, conv_state, recurrent_state, bindings):
        return {
            "provider": observe_gdn_instance(provider),
            "dispatch_metadata": observe_gdn_metadata(metadata),
            "state": observe_gdn_states(
                conv_state=conv_state, recurrent_state=recurrent_state, bindings=bindings
            ),
            "enforcement_ready": False,
        }

    def prepare_call(self, provider, inputs, metadata):
        raise StrictProviderError(
            "unobservable_plan",
            "GDN candidate metadata does not identify an authorized native/JIT dispatch",
        )

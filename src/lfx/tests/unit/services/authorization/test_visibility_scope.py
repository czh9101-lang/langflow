"""Portable visibility-scope contracts used by database list prefilters."""

from uuid import uuid4

from lfx.services.authorization import ResourceVisibilityScope, TargetedShareSelector


def test_targeted_share_selector_widens_scope_without_materializing_resource_ids():
    user_id = uuid4()
    selector = TargetedShareSelector(
        user_id=user_id,
        resource_type="flow",
        permission_levels=("write", "admin"),
    )

    scope = ResourceVisibilityScope(targeted_share=selector)

    assert scope.targeted_share == selector
    assert scope.resource_ids == ()
    assert scope.has_cross_user_access is True


def test_empty_targeted_share_selector_does_not_claim_cross_user_access():
    scope = ResourceVisibilityScope(
        targeted_share=TargetedShareSelector(
            user_id=uuid4(),
            resource_type="flow",
            permission_levels=(),
        )
    )

    assert scope.has_cross_user_access is False

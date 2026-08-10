"""Structured DB prefilter coverage for scoped authorization grants."""

from __future__ import annotations

from uuid import uuid4

import pytest
from langflow.services.authorization.listing import (
    apply_owned_or_visible_scope_prefilter,
    resource_visible_in_scope,
    restrict_to_owned_or_visible_scope,
    visible_scope_prefilter,
)
from langflow.services.database.models.auth import AuthzShare, AuthzTeam, AuthzTeamMember
from langflow.services.database.models.flow.model import Flow
from langflow.services.database.models.folder.model import Folder
from lfx.services.authorization.base import ResourceVisibilityScope, TargetedShareSelector
from sqlmodel import select

from ._common import _StubAuthorizationService, install_authz, install_settings


class _ScopedAuthorizationService(_StubAuthorizationService):
    def __init__(self, scope: ResourceVisibilityScope | None) -> None:
        super().__init__()
        self.scope = scope

    async def get_resource_visibility(self, **kwargs) -> ResourceVisibilityScope | None:
        self.visible_calls.append(kwargs)
        return self.scope


@pytest.mark.anyio
async def test_visible_scope_prefilter_forwards_structured_scope(monkeypatch, fake_user):
    install_settings(monkeypatch, authz_enabled=True)
    workspace_id = uuid4()
    project_id = uuid4()
    scope = ResourceVisibilityScope(workspace_ids=(workspace_id,), project_ids=(project_id,))
    service = _ScopedAuthorizationService(scope)
    install_authz(monkeypatch, service)

    result = await visible_scope_prefilter(fake_user, resource_type="flow", act="read")

    assert result == scope
    assert service.visible_calls == [
        {
            "user_id": fake_user.id,
            "resource_type": "flow",
            "domain": "*",
            "act": "read",
            "context": {"is_superuser": False},
        }
    ]


@pytest.mark.anyio
async def test_visible_scope_prefilter_adapts_legacy_concrete_id_service(monkeypatch, fake_user):
    install_settings(monkeypatch, authz_enabled=True)
    visible_ids = [uuid4(), uuid4()]
    service = _StubAuthorizationService(visible_ids=visible_ids)
    install_authz(monkeypatch, service)

    result = await visible_scope_prefilter(fake_user, resource_type="flow", act="read")

    assert result == ResourceVisibilityScope(resource_ids=tuple(visible_ids))
    assert len(service.visible_calls) == 1


@pytest.mark.anyio
async def test_visible_scope_prefilter_returns_none_without_contacting_plugin_when_disabled(monkeypatch, fake_user):
    install_settings(monkeypatch, authz_enabled=False)
    scope = ResourceVisibilityScope(
        targeted_share=TargetedShareSelector(
            user_id=fake_user.id,
            resource_type="flow",
            permission_levels=("read",),
        )
    )
    service = _ScopedAuthorizationService(scope)
    install_authz(monkeypatch, service)

    result = await visible_scope_prefilter(fake_user, resource_type="flow", act="read")

    assert result is None
    assert service.visible_calls == []


def test_scope_predicate_unions_owner_explicit_workspace_and_project_grants():
    owner_id = uuid4()
    scope = ResourceVisibilityScope(
        resource_ids=(uuid4(),),
        workspace_ids=(uuid4(),),
        project_ids=(uuid4(),),
    )

    constrained = restrict_to_owned_or_visible_scope(
        select(Flow),
        id_column=Flow.id,
        owner_clause=Flow.user_id == owner_id,
        workspace_column=Flow.workspace_id,
        project_column=Flow.folder_id,
        visibility=scope,
    )

    sql = str(constrained)
    assert "flow.user_id =" in sql
    assert "flow.id IN" in sql
    assert "flow.workspace_id IN" in sql
    assert "flow.folder_id IN" in sql
    assert sql.count(" OR ") == 3


def test_global_scope_does_not_emit_an_unbounded_id_list():
    constrained = restrict_to_owned_or_visible_scope(
        select(Flow),
        id_column=Flow.id,
        owner_clause=Flow.user_id == uuid4(),
        workspace_column=Flow.workspace_id,
        project_column=Flow.folder_id,
        visibility=ResourceVisibilityScope(all_resources=True),
    )

    sql = str(constrained)
    assert "flow.id IN" not in sql
    assert "flow.user_id =" not in sql


async def test_targeted_share_selector_uses_database_native_direct_and_active_team_predicates(async_session):
    owner_id = uuid4()
    target_user_id = uuid4()
    other_user_id = uuid4()
    active_team = AuthzTeam(team_name="Active", adom_name=f"active-{uuid4()}", is_active=True)
    inactive_team = AuthzTeam(team_name="Inactive", adom_name=f"inactive-{uuid4()}", is_active=False)
    owned = Flow(name="owned", user_id=owner_id)
    direct_write = Flow(name="direct write", user_id=other_user_id)
    direct_read = Flow(name="direct read", user_id=other_user_id)
    active_team_admin = Flow(name="active team admin", user_id=other_user_id)
    inactive_team_admin = Flow(name="inactive team admin", user_id=other_user_id)
    wrong_resource_type = Flow(name="wrong resource type", user_id=other_user_id)
    public_admin = Flow(name="public admin", user_id=other_user_id)
    async_session.add_all(
        [
            active_team,
            inactive_team,
            owned,
            direct_write,
            direct_read,
            active_team_admin,
            inactive_team_admin,
            wrong_resource_type,
            public_admin,
        ]
    )
    await async_session.flush()
    async_session.add_all(
        [
            AuthzTeamMember(team_id=active_team.id, user_id=target_user_id),
            AuthzTeamMember(team_id=inactive_team.id, user_id=target_user_id),
            AuthzShare(
                resource_type="flow",
                resource_id=direct_write.id,
                scope="user",
                target_id=target_user_id,
                permission_level="write",
            ),
            AuthzShare(
                resource_type="flow",
                resource_id=direct_read.id,
                scope="user",
                target_id=target_user_id,
                permission_level="read",
            ),
            AuthzShare(
                resource_type="flow",
                resource_id=active_team_admin.id,
                scope="team",
                target_id=active_team.id,
                permission_level="admin",
            ),
            AuthzShare(
                resource_type="flow",
                resource_id=inactive_team_admin.id,
                scope="team",
                target_id=inactive_team.id,
                permission_level="admin",
            ),
            AuthzShare(
                resource_type="deployment",
                resource_id=wrong_resource_type.id,
                scope="user",
                target_id=target_user_id,
                permission_level="admin",
            ),
            AuthzShare(
                resource_type="flow",
                resource_id=public_admin.id,
                scope="public",
                target_id=None,
                permission_level="admin",
            ),
        ]
    )
    await async_session.commit()

    visibility = ResourceVisibilityScope(
        targeted_share=TargetedShareSelector(
            user_id=target_user_id,
            resource_type="flow",
            # Enterprise maps WRITE to shares at WRITE or ADMIN level.
            permission_levels=("write", "admin"),
        )
    )
    stmt = restrict_to_owned_or_visible_scope(
        select(Flow),
        id_column=Flow.id,
        owner_clause=Flow.user_id == owner_id,
        visibility=visibility,
    )
    rows = list((await async_session.exec(stmt)).all())

    assert {row.id for row in rows} == {owned.id, direct_write.id, active_team_admin.id}
    sql = str(stmt)
    assert "EXISTS" in sql
    assert "authz_share.resource_id = flow.id" in sql
    assert "flow.id IN" not in sql


async def test_targeted_share_remains_additive_inside_reserved_global_project_exclusion(async_session):
    owner_id = uuid4()
    target_user_id = uuid4()
    other_user_id = uuid4()
    ordinary_project = Folder(name="ordinary")
    reserved_project = Folder(name="reserved")
    ordinary = Flow(name="ordinary", user_id=other_user_id, folder_id=ordinary_project.id)
    hidden_reserved = Flow(name="hidden reserved", user_id=other_user_id, folder_id=reserved_project.id)
    owned_reserved = Flow(name="owned reserved", user_id=owner_id, folder_id=reserved_project.id)
    shared_reserved = Flow(name="shared reserved", user_id=other_user_id, folder_id=reserved_project.id)
    async_session.add_all(
        [ordinary_project, reserved_project, ordinary, hidden_reserved, owned_reserved, shared_reserved]
    )
    await async_session.flush()
    async_session.add(
        AuthzShare(
            resource_type="flow",
            resource_id=shared_reserved.id,
            scope="user",
            target_id=target_user_id,
            permission_level="read",
        )
    )
    await async_session.commit()

    visibility = ResourceVisibilityScope(
        all_resources=True,
        excluded_global_project_ids=(reserved_project.id,),
        targeted_share=TargetedShareSelector(
            user_id=target_user_id,
            resource_type="flow",
            permission_levels=("read", "write", "execute", "admin"),
        ),
    )
    stmt = restrict_to_owned_or_visible_scope(
        select(Flow),
        id_column=Flow.id,
        owner_clause=Flow.user_id == owner_id,
        project_column=Flow.folder_id,
        visibility=visibility,
    )
    rows = list((await async_session.exec(stmt)).all())

    assert {row.id for row in rows} == {ordinary.id, owned_reserved.id, shared_reserved.id}


@pytest.mark.anyio
async def test_targeted_share_selector_honors_scoped_api_key_owner_override(monkeypatch, async_session):
    from langflow.services.authorization import listing as authz_listing

    async def _override_off() -> bool:
        return False

    monkeypatch.setattr(authz_listing, "should_apply_owner_override", _override_off)
    owner_id = uuid4()
    target_user_id = uuid4()
    owned = Flow(name="owned", user_id=owner_id)
    shared = Flow(name="shared", user_id=uuid4())
    async_session.add_all([owned, shared])
    await async_session.flush()
    async_session.add(
        AuthzShare(
            resource_type="flow",
            resource_id=shared.id,
            scope="user",
            target_id=target_user_id,
            permission_level="read",
        )
    )
    await async_session.commit()

    stmt = await apply_owned_or_visible_scope_prefilter(
        select(Flow),
        id_column=Flow.id,
        owner_clause=Flow.user_id == owner_id,
        visibility=ResourceVisibilityScope(
            targeted_share=TargetedShareSelector(
                user_id=target_user_id,
                resource_type="flow",
                permission_levels=("read",),
            )
        ),
    )
    rows = list((await async_session.exec(stmt)).all())

    assert [row.id for row in rows] == [shared.id]


async def test_global_scope_excludes_reserved_projects_but_keeps_owner_share_and_folderless(async_session):
    owner_id = uuid4()
    other_owner_id = uuid4()
    ordinary_project_id = uuid4()
    excluded_project_id = uuid4()
    ordinary_flow = Flow(name="ordinary", user_id=other_owner_id, folder_id=ordinary_project_id)
    excluded_flow = Flow(name="excluded", user_id=other_owner_id, folder_id=excluded_project_id)
    owned_excluded_flow = Flow(name="owned excluded", user_id=owner_id, folder_id=excluded_project_id)
    shared_excluded_flow = Flow(name="shared excluded", user_id=other_owner_id, folder_id=excluded_project_id)
    folderless_flow = Flow(name="folderless", user_id=other_owner_id, folder_id=None)
    async_session.add_all(
        [
            Folder(id=ordinary_project_id, name="Ordinary project"),
            Folder(id=excluded_project_id, name="Reserved project"),
            ordinary_flow,
            excluded_flow,
            owned_excluded_flow,
            shared_excluded_flow,
            folderless_flow,
        ]
    )
    await async_session.commit()

    scope = ResourceVisibilityScope(
        all_resources=True,
        resource_ids=(shared_excluded_flow.id,),
        excluded_global_project_ids=(excluded_project_id,),
    )
    stmt = restrict_to_owned_or_visible_scope(
        select(Flow),
        id_column=Flow.id,
        owner_clause=Flow.user_id == owner_id,
        workspace_column=Flow.workspace_id,
        project_column=Flow.folder_id,
        visibility=scope,
    )
    rows = list((await async_session.exec(stmt)).all())

    assert {row.id for row in rows} == {
        ordinary_flow.id,
        owned_excluded_flow.id,
        shared_excluded_flow.id,
        folderless_flow.id,
    }
    assert resource_visible_in_scope(
        resource_id=ordinary_flow.id,
        project_id=ordinary_project_id,
        visibility=scope,
    )
    assert not resource_visible_in_scope(
        resource_id=excluded_flow.id,
        project_id=excluded_project_id,
        visibility=scope,
    )
    assert resource_visible_in_scope(
        resource_id=shared_excluded_flow.id,
        project_id=excluded_project_id,
        visibility=scope,
    )
    assert resource_visible_in_scope(
        resource_id=folderless_flow.id,
        project_id=None,
        visibility=scope,
    )


def test_unassigned_workspace_scope_uses_a_compact_null_predicate():
    excluded_project_id = uuid4()
    scope = ResourceVisibilityScope(
        include_unassigned_workspace=True,
        excluded_workspace_project_ids=(excluded_project_id,),
    )
    constrained = restrict_to_owned_or_visible_scope(
        select(Flow),
        id_column=Flow.id,
        owner_clause=Flow.user_id == uuid4(),
        workspace_column=Flow.workspace_id,
        project_column=Flow.folder_id,
        visibility=scope,
    )

    sql = str(constrained)
    assert "flow.workspace_id IS NULL" in sql
    assert "flow.folder_id IS NOT NULL" in sql
    assert "flow.folder_id NOT IN" in sql
    assert not resource_visible_in_scope(
        resource_id=uuid4(),
        workspace_id=None,
        visibility=scope,
    )
    assert resource_visible_in_scope(
        resource_id=uuid4(),
        workspace_id=None,
        project_id=uuid4(),
        visibility=scope,
    )
    assert not resource_visible_in_scope(
        resource_id=uuid4(),
        workspace_id=uuid4(),
        visibility=scope,
    )
    assert not resource_visible_in_scope(
        resource_id=uuid4(),
        workspace_id=None,
        project_id=excluded_project_id,
        visibility=scope,
    )


async def test_workspace_scope_sql_matches_in_memory_for_project_nulls_and_exclusions(async_session):
    owner_id = uuid4()
    ordinary_project_id = uuid4()
    excluded_project_id = uuid4()
    explicit_workspace_id = uuid4()
    ordinary_default_flow = Flow(name="ordinary default", folder_id=ordinary_project_id, workspace_id=None)
    excluded_default_flow = Flow(name="excluded default", folder_id=excluded_project_id, workspace_id=None)
    folderless_default_flow = Flow(name="folderless default", folder_id=None, workspace_id=None)
    explicit_workspace_flow = Flow(name="workspace only", folder_id=None, workspace_id=explicit_workspace_id)
    excluded_explicit_flow = Flow(
        name="excluded explicit",
        folder_id=excluded_project_id,
        workspace_id=explicit_workspace_id,
    )
    async_session.add_all(
        [
            Folder(id=ordinary_project_id, name="Ordinary project"),
            Folder(id=excluded_project_id, name="Reserved project", workspace_id=explicit_workspace_id),
            ordinary_default_flow,
            excluded_default_flow,
            folderless_default_flow,
            explicit_workspace_flow,
            excluded_explicit_flow,
        ]
    )
    await async_session.commit()

    default_scope = ResourceVisibilityScope(
        include_unassigned_workspace=True,
        excluded_workspace_project_ids=(excluded_project_id,),
    )
    default_stmt = restrict_to_owned_or_visible_scope(
        select(Flow),
        id_column=Flow.id,
        owner_clause=Flow.user_id == owner_id,
        workspace_column=Flow.workspace_id,
        project_column=Flow.folder_id,
        visibility=default_scope,
    )
    default_rows = list((await async_session.exec(default_stmt)).all())
    assert [row.id for row in default_rows] == [ordinary_default_flow.id]

    explicit_scope = ResourceVisibilityScope(
        workspace_ids=(explicit_workspace_id,),
        excluded_workspace_project_ids=(excluded_project_id,),
    )
    explicit_stmt = restrict_to_owned_or_visible_scope(
        select(Flow),
        id_column=Flow.id,
        owner_clause=Flow.user_id == owner_id,
        workspace_column=Flow.workspace_id,
        project_column=Flow.folder_id,
        visibility=explicit_scope,
    )
    explicit_rows = list((await async_session.exec(explicit_stmt)).all())
    assert [row.id for row in explicit_rows] == [explicit_workspace_flow.id]

    cases = [
        (ordinary_default_flow, default_scope, True),
        (excluded_default_flow, default_scope, False),
        (folderless_default_flow, default_scope, False),
        (explicit_workspace_flow, explicit_scope, True),
        (excluded_explicit_flow, explicit_scope, False),
    ]
    for flow, scope, expected in cases:
        assert (
            resource_visible_in_scope(
                resource_id=flow.id,
                workspace_id=flow.workspace_id,
                project_id=flow.folder_id,
                visibility=scope,
            )
            is expected
        )


def test_scope_reports_cross_user_access_without_resource_enumeration():
    assert ResourceVisibilityScope().has_cross_user_access is False
    assert ResourceVisibilityScope(all_resources=True).has_cross_user_access is True
    assert ResourceVisibilityScope(workspace_ids=(uuid4(),)).has_cross_user_access is True
    assert ResourceVisibilityScope(include_unassigned_workspace=True).has_cross_user_access is True

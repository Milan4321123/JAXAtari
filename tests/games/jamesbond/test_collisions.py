import jax
import jax.numpy as jnp

from jaxatari.games.jax_jamesbond import JaxJamesBond


def _reset_env():
    env = JaxJamesBond()
    _, state = env.reset(jax.random.PRNGKey(0))
    return env, state


def test_player_bullet_can_hit_diamond():
    env, state = _reset_env()
    state = state.replace(
        player_bullet_active=jnp.array(True, dtype=jnp.bool_),
        player_bullet_x=jnp.array(60, dtype=jnp.int32),
        player_bullet_y=jnp.array(60, dtype=jnp.int32),
        diamond_x=state.diamond_x.at[0].set(60.0),
        diamond_y=state.diamond_y.at[0].set(60.0),
        diamond_active=state.diamond_active.at[0].set(True),
    )

    _, next_state, reward, _, info = env.step(state, jnp.array(0, dtype=jnp.int32))

    assert not bool(next_state.diamond_active[0])
    assert not bool(next_state.player_bullet_active)
    assert int(next_state.score) == env.consts.SCORE_DIAMOND
    assert float(reward) == env.consts.REWARD_DIAMOND
    assert bool(info.collected_diamond)


def test_generic_bullet_can_hit_diamond():
    env, state = _reset_env()
    state = state.replace(
        bullet_x=state.bullet_x.at[0].set(70.0),
        bullet_y=state.bullet_y.at[0].set(64.0),
        bullet_active=state.bullet_active.at[0].set(True),
        diamond_x=state.diamond_x.at[0].set(70.0),
        diamond_y=state.diamond_y.at[0].set(64.0),
        diamond_active=state.diamond_active.at[0].set(True),
    )

    _, next_state, reward, _, info = env.step(state, jnp.array(0, dtype=jnp.int32))

    assert not bool(next_state.diamond_active[0])
    assert not bool(next_state.bullet_active[0])
    assert int(next_state.score) == env.consts.SCORE_DIAMOND
    assert float(reward) == env.consts.REWARD_DIAMOND
    assert bool(info.collected_diamond)


def test_player_bullet_can_hit_enemy():
    env, state = _reset_env()
    state = state.replace(
        player_bullet_active=jnp.array(True, dtype=jnp.bool_),
        player_bullet_x=jnp.array(80, dtype=jnp.int32),
        player_bullet_y=jnp.array(72, dtype=jnp.int32),
        enemy_x=state.enemy_x.at[0].set(80.0),
        enemy_y=state.enemy_y.at[0].set(72.0),
        enemy_active=state.enemy_active.at[0].set(True),
    )

    _, next_state, reward, _, info = env.step(state, jnp.array(0, dtype=jnp.int32))

    assert not bool(next_state.enemy_active[0])
    assert not bool(next_state.player_bullet_active)
    assert int(next_state.score) == env.consts.SCORE_ENEMY
    assert float(reward) == env.consts.REWARD_ENEMY
    assert bool(info.hit_enemy)

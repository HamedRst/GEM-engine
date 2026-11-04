"""Exploration GUI package; pure data builder can run without NiceGUI installed."""
__all__ = ['register_exploration_routes', 'render_exploration_tab']

def __getattr__(name):
    if name in __all__:
        from .component import register_exploration_routes, render_exploration_tab
        return {'register_exploration_routes': register_exploration_routes,
                'render_exploration_tab': render_exploration_tab}[name]
    raise AttributeError(name)

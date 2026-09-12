# Home Assistant Plugin Capability Manifest Specification

## Overview

An optional HA custom integration that exposes a REST endpoint providing
semantic capability metadata to the Brain chatbot system. This bridges
HA's domain naming (e.g., `light`, `media_player`) with the Brain's
trained intent domain prefixes (e.g., `smarthome`, `music`).

Without this plugin, the system uses standard HA APIs (`/api/services`,
`/api/states`) and the `ha_domain_types.json` config for basic mapping.
The plugin enables richer integration.

## Endpoint

```
GET /api/chatbot/capabilities
```

Authorization: Standard HA Bearer token (same as other HA REST APIs).

## Response Format

```json
{
  "version": 1,
  "instance_name": "My Home",
  "domain_mappings": {
    "light": "smarthome",
    "switch": "smarthome",
    "cover": "smarthome",
    "fan": "smarthome",
    "climate": "smarthome",
    "media_player": "music",
    "automation": "smarthome",
    "timer": "timer",
    "calendar": "calendar",
    "alarm_control_panel": "alarm"
  },
  "areas": [
    {"id": "living_room", "name": "Living Room"},
    {"id": "bedroom", "name": "Bedroom"},
    {"id": "office", "name": "Office"}
  ],
  "entity_aliases": {
    "light.office_ceiling": ["office light", "office lamp", "ceiling light"],
    "media_player.living_room": ["speaker", "living room speaker", "music"]
  },
  "custom_actions": [
    {
      "name": "goodnight",
      "description": "Turn off all lights and lock doors",
      "trigger": {"domain": "script", "service": "turn_on", "entity_id": "script.goodnight"}
    },
    {
      "name": "movie_mode",
      "description": "Dim lights and turn on the TV",
      "trigger": {"domain": "scene", "service": "turn_on", "entity_id": "scene.movie_mode"}
    }
  ],
  "available_features": {
    "media_control": true,
    "climate_control": true,
    "lighting": true,
    "locks": true,
    "cameras": false,
    "energy_monitoring": true,
    "voice_assistants": false
  }
}
```

## Field Descriptions

### domain_mappings

Maps HA integration domains to Brain intent domain prefixes. This tells
the Brain which of its trained intent classifiers should route to HA.

The Brain's classifier produces intents like `smarthome.switch` or
`music.play`. The domain prefix (`smarthome`, `music`) determines
which service handles it. This mapping connects HA's internal domain
names to those trained prefixes.

### areas

List of configured areas/rooms in the HA instance. Used to:
- Register as Gazetteer entries (type: "room") for NER
- Help resolve ambiguous entity references ("turn on the light" -> which room?)

### entity_aliases

Optional user-defined aliases for entities. Registered in the Gazetteer
alongside the entity's friendly_name so users can refer to devices by
multiple names.

### custom_actions

Named macros/scenes/scripts that can be triggered by name. These are
registered as special entities in the Gazetteer and can be invoked when
the user says things like "activate movie mode" or "run goodnight routine."

### available_features

Boolean flags indicating which broad feature categories this HA instance
supports. Allows the Brain to:
- Avoid routing intents to HA when the feature isn't available
- Provide appropriate "not supported" responses

## HA Integration Implementation

The plugin is a standard HA custom integration:

```
custom_components/
  chatbot_capabilities/
    __init__.py
    manifest.json
    const.py
```

### manifest.json

```json
{
  "domain": "chatbot_capabilities",
  "name": "Chatbot Capabilities",
  "version": "1.0.0",
  "documentation": "https://github.com/your-repo",
  "dependencies": [],
  "codeowners": [],
  "config_flow": false,
  "integration_type": "service"
}
```

### __init__.py

Registers a single REST endpoint that introspects the HA instance
and builds the capability manifest dynamically from:
- `hass.states.async_all()` for entities
- `hass.services.async_services()` for available services
- `hass.data["area_registry"]` for areas
- User-configured aliases from `configuration.yaml`

## Brain Consumption

When the `CapabilityRegistry` detects this endpoint is available
(by attempting a GET and checking for a valid response), it uses
the richer manifest data instead of inferring mappings from the
standard APIs alone.

The consumption priority is:
1. Plugin manifest (if available) -- richest data
2. Standard `/api/services` + `/api/states` -- always available
3. Static `ha_domain_types.json` fallback -- minimal offline mapping

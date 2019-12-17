#pragma once

#include "hdtSkyrimMesh.h"
#include "hdtSkinnedMesh\hdtSkinnedMeshWorld.h"
#include "IEventListener.h"
#include "HookEvents.h"

#include <atomic>

namespace hdt
{
	static float TIME_TICK = 1 / 60.f;

	class SkyrimPhysicsWorld : public SkinnedMeshWorld, public IEventListener<FrameEvent>, public IEventListener<ShutdownEvent>
	{
	public:

		static SkyrimPhysicsWorld* get();

		void doUpdate(float delta);
		void updateActiveState();

		virtual void addSkinnedMeshSystem(SkinnedMeshSystem* system) override;
		virtual void removeSkinnedMeshSystem(SkinnedMeshSystem* system) override;
		void removeSystemByNode(void* root);

		void resetSystems();

		virtual void onEvent(const FrameEvent& e) override;
		virtual void onEvent(const ShutdownEvent& e) override;

		inline void suspend(bool loading = false) { m_suspended = true; m_loading = loading; }
		inline void resume() {
			m_suspended = false;
			if (m_loading)
			{
				resetSystems();
				m_loading = false;
			}
		}

		btVector3 applyTranslationOffset();
		void restoreTranslationOffset(const btVector3&);

	private:

		SkyrimPhysicsWorld(void);
		~SkyrimPhysicsWorld(void);

		std::mutex m_lock;

		std::atomic_bool m_suspended;
		std::atomic_bool m_loading;
		float m_averageInterval;
		float m_accumulatedInterval;
	};
}

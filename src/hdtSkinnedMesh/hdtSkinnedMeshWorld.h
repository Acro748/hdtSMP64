#pragma once

#include "hdtSkinnedMeshSystem.h"
#include "hdtSkyrimSystem.h"
#include <BulletCollision/CollisionDispatch/btSimulationIslandManager.h>
#include <BulletDynamics/Dynamics/btDiscreteDynamicsWorldMt.h>

namespace hdt
{
	class SkinnedMeshWorld : protected btDiscreteDynamicsWorldMt
	{
	public:
		SkinnedMeshWorld();
		~SkinnedMeshWorld();

		virtual void addSkinnedMeshSystem(SkinnedMeshSystem* system);
		virtual void removeSkinnedMeshSystem(SkinnedMeshSystem* system);

		int stepSimulation(btScalar remainingTimeStep, int maxSubSteps = 1,
			btScalar fixedTimeStep = btScalar(1.) / btScalar(60.)) override;

		btVector3& getWind() { return m_windSpeed; }
		const btVector3& getWind() const { return m_windSpeed; }

	protected:
		// Contains grouped systems that share the same actor
		std::vector<SkyrimSystem*> m_sorted;
		std::vector<float> m_timeSteps;

		void resetTransformsToOriginal()
		{
			for (int i = 0; i < m_systems.size(); ++i) m_systems[i]->resetTransformsToOriginal();
		}

		void readTransform(float timeStep)
		{
			const size_t n = m_systems.size();
			if (n == 0)
				return;

			m_sorted.resize(n);
			for (size_t i = 0; i < n; ++i)
				m_sorted[i] = static_cast<SkyrimSystem*>(m_systems[i].get());

			std::ranges::sort(m_sorted, std::less<>{}, [](const SkyrimSystem* sys) {
				return sys->m_skeleton.get();
			});

			m_timeSteps.resize(n);

			// Loop over the systems to find all the ones connected to one body, and only call processSkeletonRoot once
			size_t i = 0;
			while (i < n) {
				SkyrimSystem* first = m_sorted[i];
				float ts = first->processSkeletonRoot(timeStep);
				void* key = first->m_skeleton.get();
				m_timeSteps[i] = ts;

				for (size_t j = i + 1; j < n && m_sorted[j]->m_skeleton.get() == key; ++j) {
					m_sorted[j]->m_lastRootRotation = first->m_lastRootRotation;
					m_sorted[j]->m_oldRoot = first->m_oldRoot;
					m_timeSteps[j] = ts;
					i = j;
				}
				++i;
			}

			concurrency::parallel_for(size_t{ 0 }, n,
				[this](size_t i) {
					m_sorted[i]->readTransform(m_timeSteps[i]);
				});
		}

		void writeTransform()
		{
			for (int i = 0; i < m_systems.size(); ++i) m_systems[i]->writeTransform();
		}

		void applyGravity() override;
		void applyWind();

		void predictUnconstraintMotion(btScalar timeStep) override;
		void integrateTransforms(btScalar timeStep) override;
		void performDiscreteCollisionDetection() override;
		void calculateSimulationIslands() override;
		void solveConstraints(btContactSolverInfo& solverInfo) override;

		std::vector<RE::BSTSmartPointer<SkinnedMeshSystem>> m_systems;

		btVector3 m_windSpeed;  // world windspeed

	private:
		std::vector<SkinnedMeshBody*> _bodies;
		std::vector<SkinnedMeshShape*> _shapes;
	};
}

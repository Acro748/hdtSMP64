#pragma once

#include "skse64/GameEvents.h"

#include "HookEvents.h"
#include "IEventListener.h"

namespace hdt
{
	class EventDebugLogger
		: public IEventListener<ArmorAttachEvent>
		  , public BSTEventSink<TESCellAttachDetachEvent>
		  , public BSTEventSink<TESMoveAttachDetachEvent>
	{
	protected:
		EventResult ReceiveEvent(TESCellAttachDetachEvent* evn,
		                         EventDispatcher<TESCellAttachDetachEvent>* dispatcher) override;
		EventResult ReceiveEvent(TESMoveAttachDetachEvent* evn,
		                         EventDispatcher<TESMoveAttachDetachEvent>* dispatcher) override;

		void onEvent(const ArmorAttachEvent&) override;
	};
}

#include "hdtDefaultBBP.h"

#include "XmlReader.h"

#include <clocale>

#include "../hdtSSEUtils/NetImmerseUtils.h"

#include <algorithm>

namespace hdt
{
	DefaultBBP* DefaultBBP::instance()
	{
		static DefaultBBP s;
		return &s;
	}

	DefaultBBP::PhysicsFile DefaultBBP::scanBBP(NiNode* scan)
	{
		for (int i = 0; i < scan->m_extraDataLen; ++i)
		{
			auto stringData = ni_cast(scan->m_extraData[i], NiStringExtraData);
			if (stringData && !strcmp(stringData->m_pcName, "HDT Skinned Mesh Physics Object") && stringData->m_pString)
				return { stringData->m_pString, defaultNameMap(scan) };
		}

		return scanDefaultBBP(scan);
	}

	DefaultBBP::DefaultBBP()
	{
		loadDefaultBBPs();
	}

	void DefaultBBP::loadDefaultBBPs()
	{
		auto path = "SKSE/Plugins/hdtSkinnedMeshConfigs/defaultBBPs.xml";

		auto loaded = readAllFile(path);
		if (loaded.empty()) return;

		// Store original locale
		char saved_locale[32];
		strcpy_s(saved_locale, std::setlocale(LC_NUMERIC, nullptr));

		// Set locale to en_US
		std::setlocale(LC_NUMERIC, "en_US");

		XMLReader reader((uint8_t*)loaded.data(), loaded.size());

		reader.nextStartElement();
		if (reader.GetName() != "default-bbps")
			return;

		while (reader.Inspect())
		{
			if (reader.GetInspected() == Xml::Inspected::StartTag)
			{
				if (reader.GetName() == "map")
				{
					try
					{
						auto shape = reader.getAttribute("shape");
						auto file = reader.getAttribute("file");
						bbpFileList.insert(std::make_pair(shape, file));
					}
					catch (...)
					{
						_WARNING("defaultBBP(%d,%d) : invalid map", reader.GetRow(), reader.GetColumn());
					}
					reader.skipCurrentElement();
				}
				else if (reader.GetName() == "remap")
				{
					auto target = reader.getAttribute("target");
					Remap remap = { target, { } };
					while (reader.Inspect())
					{
						if (reader.GetInspected() == Xml::Inspected::StartTag)
						{
							if (reader.GetName() == "source")
							{
								int priority = 0;
								try
								{
									priority = reader.getAttributeAsInt("priority");
								}
								catch (...) {}
								auto source = reader.readText();
								remap.second.insert({ priority, source });
							}
							else
							{
								_WARNING("defaultBBP(%d,%d) : unknown element", reader.GetRow(), reader.GetColumn());
								reader.skipCurrentElement();
							}
						}
						else if (reader.GetInspected() == Xml::Inspected::EndTag)
						{
							break;
						}
					}
					remaps.push_back(remap);
				}
				else
				{
					_WARNING("defaultBBP(%d,%d) : unknown element", reader.GetRow(), reader.GetColumn());
					reader.skipCurrentElement();
				}
			}
			else if (reader.GetInspected() == Xml::Inspected::EndTag)
				break;
		}

		// Restore original locale
		std::setlocale(LC_NUMERIC, saved_locale);
	}

	DefaultBBP::PhysicsFile DefaultBBP::scanDefaultBBP(NiNode* armor)
	{
		static std::mutex s_lock;
		std::lock_guard<std::mutex> l(s_lock);

		if (bbpFileList.empty()) return { "", {} };

		auto remappedNames = DefaultBBP::instance()->getNameMap(armor);

		auto it = std::find_if(bbpFileList.begin(), bbpFileList.end(), [&](const std::pair<std::string, std::string>& e)
			{ return remappedNames.find(e.first) != remappedNames.end(); });
		return { it == bbpFileList.end() ? "" : it->second, remappedNames };
	}

	DefaultBBP::NameMap DefaultBBP::getNameMap(NiNode* armor)
	{
		auto nameMap = defaultNameMap(armor);

		for (auto remap : remaps)
		{
			auto start = std::find_if(remap.second.rbegin(), remap.second.rend(), [&](const RemapEntry& e)
			{ return nameMap.find(e.second) != nameMap.end(); });
			auto end = std::find_if(start, remap.second.rend(), [&](const RemapEntry& e)
			{ return e.first != start->first; });
			if (start != remap.second.rend())
			{
				auto& s = nameMap.insert({ remap.first, { } }).first;
				std::for_each(start, end, [&](const RemapEntry& e)
				{
					auto it = nameMap.find(e.second);
					if (it != nameMap.end())
					{
						std::for_each(it->second.begin(), it->second.end(), [&](const std::string& name)
						{
							s->second.insert(name);
						});
					}
				});
			}
		}
		return nameMap;
	}

	DefaultBBP::NameMap DefaultBBP::defaultNameMap(NiNode* armor)
	{
		std::unordered_map<std::string, std::unordered_set<std::string> > nameMap;
		for (int i = 0; i < armor->m_children.m_arrayBufLen; ++i)
		{
			if (!armor->m_children.m_data[i]) continue;
			auto tri = armor->m_children.m_data[i]->GetAsBSTriShape();
			if (!tri) continue;
			nameMap.insert({ tri->m_name, {tri->m_name} });
		}
		return nameMap;
	}
}

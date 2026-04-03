"""
MVTec数据集和对话内容加载脚本

功能：
1. 加载MVTec异常检测数据集的训练集和测试集
2. 加载对话内容（mvtec_zero_shot.json）
3. 提供便捷的数据访问接口
"""

import os
import json
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


class MVTecDataLoader:
    """MVTec数据集加载器"""
    
    # MVTec数据集的所有类别
    CLSNAMES = [
        'bottle', 'cable', 'capsule', 'carpet', 'grid',
        'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
        'tile', 'toothbrush', 'transistor', 'wood', 'zipper',
    ]
    
    def __init__(self, dataset_root: str):
        """
        初始化数据加载器
        
        Args:
            dataset_root: MVTec数据集根目录路径
        """
        self.dataset_root = dataset_root
        self.train_data = defaultdict(list)  # {class_name: [data_items]}
        self.test_data = defaultdict(list)   # {class_name: [data_items]}
        
    def load_train_data(self) -> Dict[str, List[Dict]]:
        """
        加载训练集数据
        
        Returns:
            字典，键为类别名，值为该类别下的数据项列表
        """
        print("正在加载训练集数据...")
        for cls_name in self.CLSNAMES:
            cls_dir = os.path.join(self.dataset_root, cls_name, 'train')
            if not os.path.exists(cls_dir):
                print(f"警告: 类别 {cls_name} 的训练目录不存在")
                continue
                
            # 训练集只包含正常样本（good）
            good_dir = os.path.join(cls_dir, 'good')
            if os.path.exists(good_dir):
                img_names = sorted([f for f in os.listdir(good_dir) 
                                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                
                for img_name in img_names:
                    data_item = {
                        'img_path': os.path.join(cls_name, 'train', 'good', img_name),
                        'full_img_path': os.path.join(self.dataset_root, cls_name, 'train', 'good', img_name),
                        'mask_path': None,
                        'cls_name': cls_name,
                        'specie_name': 'good',
                        'anomaly': 0,  # 0表示正常样本
                    }
                    self.train_data[cls_name].append(data_item)
                    
        train_count = sum(len(items) for items in self.train_data.values())
        print(f"训练集加载完成: 共 {train_count} 个样本（仅正常样本）")
        return dict(self.train_data)
    
    def load_test_data(self) -> Dict[str, List[Dict]]:
        """
        加载测试集数据
        
        Returns:
            字典，键为类别名，值为该类别下的数据项列表
        """
        print("正在加载测试集数据...")
        for cls_name in self.CLSNAMES:
            cls_dir = os.path.join(self.dataset_root, cls_name, 'test')
            if not os.path.exists(cls_dir):
                print(f"警告: 类别 {cls_name} 的测试目录不存在")
                continue
                
            species = [d for d in os.listdir(cls_dir) 
                      if os.path.isdir(os.path.join(cls_dir, d))]
            
            for specie in species:
                specie_dir = os.path.join(cls_dir, specie)
                is_abnormal = (specie != 'good')
                
                img_names = sorted([f for f in os.listdir(specie_dir) 
                                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                
                # 获取对应的mask路径（如果有）
                mask_dir = None
                if is_abnormal:
                    # 尝试多个可能的mask路径
                    possible_mask_dirs = [
                        os.path.join(self.dataset_root, cls_name, 'ground_truth', specie),
                        os.path.join(self.dataset_root, cls_name, 'groundtruth', specie),
                    ]
                    for md in possible_mask_dirs:
                        if os.path.exists(md):
                            mask_dir = md
                            break
                
                for img_name in img_names:
                    # 构建mask路径
                    mask_path = None
                    if mask_dir:
                        # 尝试多种mask文件名格式
                        mask_name_base = os.path.splitext(img_name)[0]
                        possible_mask_names = [
                            f"{mask_name_base}_mask.png",
                            f"{mask_name_base}.png",
                            img_name,
                        ]
                        for mask_name in possible_mask_names:
                            mask_full_path = os.path.join(mask_dir, mask_name)
                            if os.path.exists(mask_full_path):
                                mask_path = os.path.join(cls_name, 'ground_truth', specie, mask_name)
                                break
                    
                    data_item = {
                        'img_path': os.path.join(cls_name, 'test', specie, img_name),
                        'full_img_path': os.path.join(self.dataset_root, cls_name, 'test', specie, img_name),
                        'mask_path': mask_path,
                        'full_mask_path': os.path.join(self.dataset_root, mask_path) if mask_path else None,
                        'cls_name': cls_name,
                        'specie_name': specie,
                        'anomaly': 1 if is_abnormal else 0,
                    }
                    self.test_data[cls_name].append(data_item)
                    
        test_count = sum(len(items) for items in self.test_data.values())
        train_count = sum(len(items) for items in self.train_data.values())
        print(f"测试集加载完成: 共 {test_count} 个样本（正常+异常样本）")
        total_count = train_count + test_count
        print(f"总计: {total_count} 个样本（所有数据将作为训练集使用）")
        return dict(self.test_data)
    
    def get_train_samples_by_class(self, cls_name: str) -> List[Dict]:
        """获取指定类别的训练样本"""
        return self.train_data.get(cls_name, [])
    
    def get_test_samples_by_class(self, cls_name: str) -> List[Dict]:
        """获取指定类别的测试样本"""
        return self.test_data.get(cls_name, [])
    
    def get_all_train_samples(self) -> List[Dict]:
        """获取所有训练样本（包含原始训练集和测试集的所有数据）"""
        all_samples = []
        # 包含原始训练集
        for samples in self.train_data.values():
            all_samples.extend(samples)
        # 包含原始测试集（所有数据都作为训练集）
        for samples in self.test_data.values():
            all_samples.extend(samples)
        return all_samples
    
    def get_all_test_samples(self) -> List[Dict]:
        """获取所有测试样本（注意：如果所有数据都作为训练集，测试集可能为空）"""
        all_samples = []
        for samples in self.test_data.values():
            all_samples.extend(samples)
        return all_samples
    
    def extract_bbox_from_mask(self, mask_path: str) -> Optional[List[int]]:
        """
        从mask图像中提取边界框
        
        Args:
            mask_path: mask图像路径
            
        Returns:
            bbox坐标 [x1, y1, x2, y2]，如果没有异常区域则返回None
        """
        if mask_path is None or not os.path.exists(mask_path):
            return None
        
        try:
            mask = Image.open(mask_path).convert("L")
            mask_array = np.array(mask) > 0  # 转换为二值mask
            
            # 找到所有非零像素的行和列
            rows = np.any(mask_array, axis=1)
            cols = np.any(mask_array, axis=0)
            
            if not (rows.any() and cols.any()):
                return None  # 没有异常区域
            
            # 找到边界框
            y_indices = np.where(rows)[0]
            x_indices = np.where(cols)[0]
            
            y1, y2 = y_indices[0], y_indices[-1]
            x1, x2 = x_indices[0], x_indices[-1]
            
            # 返回 [x1, y1, x2, y2] 格式
            return [int(x1), int(y1), int(x2), int(y2)]
        except Exception as e:
            print(f"Error extracting bbox from {mask_path}: {e}")
            return None
    
    def convert_to_grounding_format(
        self, 
        sample: Dict, 
        conversation_data: Optional[Dict] = None,
        use_original_conversation: bool = True
    ) -> Dict:
        """
        将MVTec样本转换为Grounding格式
        
        Args:
            sample: MVTec数据样本
            conversation_data: 原始对话数据（可选）
            use_original_conversation: 是否使用原始对话作为基础
            
        Returns:
            Grounding格式的数据样本
        """
        img_path = sample['img_path']
        is_anomaly = sample['anomaly'] == 1
        cls_name = sample['cls_name']
        specie_name = sample['specie_name']
        
        # 提取bbox（如果有mask）
        bbox = None
        if is_anomaly and sample.get('full_mask_path'):
            bbox = self.extract_bbox_from_mask(sample['full_mask_path'])
        
        # 构建Grounding格式的对话
        if use_original_conversation and conversation_data:
            # 使用原始对话，但修改assistant回复为bbox格式
            conversations = conversation_data.get('conversations', [])
            
            # 查找最后一个assistant回复，替换为bbox格式
            new_conversations = []
            for conv in conversations:
                if conv.get('from') == 'gpt' or conv.get('role') == 'assistant':
                    # 替换为Grounding格式
                    if bbox is not None:
                        bbox_json = json.dumps({
                            "bbox_2d": bbox,
                            "label": "anomaly",
                            "defect_type": specie_name,
                            "class": cls_name
                        })
                        new_conversations.append({
                            "from": "gpt",
                            "value": bbox_json
                        })
                    else:
                        # 没有bbox，保持原样或返回空bbox
                        bbox_json = json.dumps({
                            "bbox_2d": None,
                            "label": "normal"
                        })
                        new_conversations.append({
                            "from": "gpt",
                            "value": bbox_json
                        })
                else:
                    # 保持用户输入，但修改为Grounding指令
                    if conv.get('from') == 'human' or conv.get('role') == 'user':
                        # 修改为Grounding指令
                        new_conversations.append({
                            "from": "human",
                            "value": "<image>\nLocate the anomaly region in this image and output the bbox coordinates in JSON format. If no anomaly is found, output {\"bbox_2d\": null, \"label\": \"normal\"}."
                        })
                    else:
                        new_conversations.append(conv)
        else:
            # 创建新的Grounding对话
            if is_anomaly and bbox is not None:
                bbox_json = json.dumps({
                    "bbox_2d": bbox,
                    "label": "anomaly",
                    "defect_type": specie_name,
                    "class": cls_name
                })
                conversations = [
                    {
                        "from": "human",
                        "value": "<image>\nLocate the anomaly region in this image and output the bbox coordinates in JSON format."
                    },
                    {
                        "from": "gpt",
                        "value": bbox_json
                    }
                ]
            else:
                # 正常样本
                bbox_json = json.dumps({
                    "bbox_2d": None,
                    "label": "normal"
                })
                conversations = [
                    {
                        "from": "human",
                        "value": "<image>\nLocate the anomaly region in this image and output the bbox coordinates in JSON format. If no anomaly is found, output {\"bbox_2d\": null, \"label\": \"normal\"}."
                    },
                    {
                        "from": "gpt",
                        "value": bbox_json
                    }
                ]
        
        # 构建Grounding格式样本
        grounding_sample = {
            "id": f"{cls_name}_{specie_name}_{os.path.basename(img_path)}",
            "image": img_path,
            "conversations": conversations,
            "metadata": {
                "anomaly": is_anomaly,
                "class": cls_name,
                "defect_type": specie_name,
                "bbox": bbox,
                "full_mask_path": sample.get("full_mask_path"),
                "source": "mvtec_anomaly_detection"
            }
        }
        
        return grounding_sample


class ConversationLoader:
    """对话内容加载器"""
    
    def __init__(self, conversation_json_path: str):
        """
        初始化对话加载器
        
        Args:
            conversation_json_path: 对话JSON文件路径
        """
        self.conversation_json_path = conversation_json_path
        self.conversations = []
        self.conversations_by_image = {}  # {image_path: conversation_data}
        
    def load_conversations(self) -> List[Dict]:
        """
        加载对话内容
        
        Returns:
            对话数据列表
        """
        print(f"正在加载对话内容: {self.conversation_json_path}")
        with open(self.conversation_json_path, 'r', encoding='utf-8') as f:
            self.conversations = json.load(f)
        
        # 建立图像路径到对话的映射
        for conv in self.conversations:
            image_path = conv.get('image', '')
            self.conversations_by_image[image_path] = conv
            
        print(f"对话内容加载完成: 共 {len(self.conversations)} 条对话")
        return self.conversations
    
    def get_conversation_by_image(self, image_path: str) -> Optional[Dict]:
        """
        根据图像路径获取对应的对话
        
        Args:
            image_path: 图像路径（可以是相对路径或完整路径）
            
        Returns:
            对话数据，如果不存在则返回None
        """
        # 尝试直接匹配
        if image_path in self.conversations_by_image:
            return self.conversations_by_image[image_path]
        
        # 尝试匹配相对路径
        for key, value in self.conversations_by_image.items():
            if image_path.endswith(key) or key.endswith(image_path):
                return value
        
        return None
    
    def get_conversations_by_class(self, cls_name: str) -> List[Dict]:
        """
        获取指定类别的所有对话
        
        Args:
            cls_name: 类别名称
            
        Returns:
            该类别下的对话列表
        """
        filtered_convs = []
        for conv in self.conversations:
            image_path = conv.get('image', '')
            if f'/{cls_name}/' in image_path:
                filtered_convs.append(conv)
        return filtered_convs
    
    def get_conversations_by_anomaly(self, is_anomaly: bool) -> List[Dict]:
        """
        根据是否异常获取对话
        
        Args:
            is_anomaly: True表示异常样本，False表示正常样本
            
        Returns:
            对话列表
        """
        filtered_convs = []
        for conv in self.conversations:
            metadata = conv.get('metadata', {})
            if metadata.get('anomaly', False) == is_anomaly:
                filtered_convs.append(conv)
        return filtered_convs


class MVTecDataManager:
    """MVTec数据集和对话内容的统一管理器"""
    
    def __init__(self, dataset_root: str, conversation_json_path: str):
        """
        初始化数据管理器
        
        Args:
            dataset_root: MVTec数据集根目录路径
            conversation_json_path: 对话JSON文件路径
        """
        self.dataset_loader = MVTecDataLoader(dataset_root)
        self.conversation_loader = ConversationLoader(conversation_json_path)
        
    def load_all(self):
        """加载所有数据"""
        self.dataset_loader.load_train_data()
        self.dataset_loader.load_test_data()
        self.conversation_loader.load_conversations()
    
    def get_sample_with_conversation(self, cls_name: str, mode: str = 'test', 
                                     index: int = 0, grounding_format: bool = False) -> Optional[Dict]:
        """
        获取指定样本及其对应的对话
        
        Args:
            cls_name: 类别名称
            mode: 'train' 或 'test'
            index: 样本索引
            grounding_format: 是否返回Grounding格式
            
        Returns:
            包含样本数据和对话的字典
        """
        if mode == 'train':
            samples = self.dataset_loader.get_train_samples_by_class(cls_name)
        else:
            samples = self.dataset_loader.get_test_samples_by_class(cls_name)
        
        if index >= len(samples):
            return None
        
        sample = samples[index]
        
        # 查找对应的对话
        img_path = sample['img_path']
        conversation = self.conversation_loader.get_conversation_by_image(img_path)
        
        if grounding_format:
            # 转换为Grounding格式
            return self.dataset_loader.convert_to_grounding_format(
                sample, 
                conversation,
                use_original_conversation=True
            )
        else:
            result = sample.copy()
            result['conversation'] = conversation
            return result
    
    def get_all_grounding_samples(self, mode: str = 'test') -> List[Dict]:
        """
        获取所有Grounding格式的样本
        
        Args:
            mode: 'train' 或 'test'
            
        Returns:
            Grounding格式的样本列表
        """
        from tqdm import tqdm
        
        if mode == 'train':
            samples = self.dataset_loader.get_all_train_samples()
        else:
            samples = self.dataset_loader.get_all_test_samples()
        
        print(f"\n正在转换 {len(samples)} 个样本为Grounding格式...")
        grounding_samples = []
        
        for sample in tqdm(samples, desc=f"转换{mode}集", unit="样本"):
            img_path = sample['img_path']
            conversation = self.conversation_loader.get_conversation_by_image(img_path)
            grounding_sample = self.dataset_loader.convert_to_grounding_format(
                sample,
                conversation,
                use_original_conversation=True
            )
            grounding_samples.append(grounding_sample)
        
        return grounding_samples
    
    def save_grounding_dataset(self, output_path: str, mode: str = 'test'):
        """
        保存Grounding格式的数据集到JSON文件
        
        Args:
            output_path: 输出文件路径
            mode: 'train' 或 'test'
        """
        grounding_samples = self.get_all_grounding_samples(mode)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(grounding_samples, f, indent=2, ensure_ascii=False)
        
        print(f"已保存 {len(grounding_samples)} 个Grounding格式样本到 {output_path}")
        return grounding_samples


def main():
    """示例用法"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MVTec数据加载和Grounding格式转换")
    parser.add_argument("--mode", type=str, default="view", choices=["view", "convert"],
                       help="模式: view查看数据, convert转换为Grounding格式")
    parser.add_argument("--output_path", type=str, default=None,
                       help="Grounding格式输出路径（convert模式需要）")
    parser.add_argument("--dataset_mode", type=str, default="test", choices=["train", "test"],
                       help="数据集模式: train或test")
    
    args = parser.parse_args()
    
    # 数据集路径
    dataset_root = '/data2/zlt/anomaly_detection_llm/datasets/mvtec_anomaly_detection'
    conversation_json_path = '/data2/zlt/anomaly_detection_llm/datasets/mvtec_zero_shot.json'
    
    # 创建数据管理器
    manager = MVTecDataManager(dataset_root, conversation_json_path)
    
    # 加载所有数据
    print("=" * 60)
    print("开始加载数据...")
    print("=" * 60)
    manager.load_all()
    
    if args.mode == "convert":
        # 转换为Grounding格式并保存
        if args.output_path is None:
            args.output_path = f"/data2/zlt/anomaly_detection_llm/datasets/mvtec_grounding_{args.dataset_mode}.json"
        
        print(f"\n{'='*60}")
        print(f"转换为Grounding格式 [{args.dataset_mode}]")
        print(f"{'='*60}")
        grounding_samples = manager.save_grounding_dataset(args.output_path, mode=args.dataset_mode)
        
        # 显示统计信息
        anomaly_count = sum(1 for s in grounding_samples if s.get('metadata', {}).get('anomaly', False))
        normal_count = len(grounding_samples) - anomaly_count
        bbox_count = sum(1 for s in grounding_samples if s.get('metadata', {}).get('bbox') is not None)
        
        print(f"\nGrounding数据集统计:")
        print(f"  总样本数: {len(grounding_samples)}")
        print(f"  异常样本: {anomaly_count}")
        print(f"  正常样本: {normal_count}")
        print(f"  有bbox的样本: {bbox_count}")
        print(f"\n已保存到: {args.output_path}")
        
    else:
        # 显示统计信息
        print("\n" + "=" * 60)
        print("数据统计信息:")
        print("=" * 60)
        
        train_data = manager.dataset_loader.train_data
        test_data = manager.dataset_loader.test_data
        
        print("\n训练集统计:")
        for cls_name in manager.dataset_loader.CLSNAMES:
            train_count = len(train_data.get(cls_name, []))
            print(f"  {cls_name}: {train_count} 个样本")
        
        print("\n测试集统计:")
        for cls_name in manager.dataset_loader.CLSNAMES:
            test_samples = test_data.get(cls_name, [])
            normal_count = sum(1 for s in test_samples if s['anomaly'] == 0)
            anomaly_count = sum(1 for s in test_samples if s['anomaly'] == 1)
            print(f"  {cls_name}: 正常={normal_count}, 异常={anomaly_count}, 总计={len(test_samples)}")
        
        print(f"\n对话数据统计:")
        print(f"  总对话数: {len(manager.conversation_loader.conversations)}")
        anomaly_convs = manager.conversation_loader.get_conversations_by_anomaly(True)
        normal_convs = manager.conversation_loader.get_conversations_by_anomaly(False)
        print(f"  异常样本对话: {len(anomaly_convs)}")
        print(f"  正常样本对话: {len(normal_convs)}")
        
        # 示例：获取一个样本及其对话（Grounding格式）
        print("\n" + "=" * 60)
        print("示例：获取Grounding格式样本")
        print("=" * 60)
        
        sample = manager.get_sample_with_conversation('bottle', mode='test', index=0, grounding_format=True)
        if sample:
            print(f"\n样本信息:")
            print(f"  ID: {sample.get('id', 'N/A')}")
            print(f"  图像路径: {sample.get('image', 'N/A')}")
            metadata = sample.get('metadata', {})
            print(f"  是否异常: {metadata.get('anomaly', 'N/A')}")
            print(f"  类别: {metadata.get('class', 'N/A')}")
            print(f"  缺陷类型: {metadata.get('defect_type', 'N/A')}")
            print(f"  Bbox: {metadata.get('bbox', 'N/A')}")
            
            if sample.get('conversations'):
                print(f"\n对话信息:")
                for i, conv in enumerate(sample['conversations']):
                    print(f"  轮次 {i+1}:")
                    print(f"    {conv.get('from', 'N/A')}: {conv.get('value', '')[:150]}...")


if __name__ == '__main__':
    main()

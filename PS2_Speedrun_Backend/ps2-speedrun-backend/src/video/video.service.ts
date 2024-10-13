import { Injectable } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Model } from 'mongoose';
import { Video, VideoDocument } from './schema/video.schema';
import { CreateVideoDto } from './dto/create-video.dto';

@Injectable()
export class VideoService {
  constructor(
    @InjectModel(Video.name) private videoModel: Model<VideoDocument>,
  ) {}

  async create(createVideoDto: CreateVideoDto): Promise<Video> {
    const createdVideo = new this.videoModel(createVideoDto);
    return createdVideo.save();
  }

  async findAll(): Promise<Video[]> {
    return this.videoModel.find().exec();
  }
}
